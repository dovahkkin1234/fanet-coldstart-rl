"""
apply_pre_m4_fixes_v1.py  —  THE single pre-M4 fix pass.

Applies every agreed correction in ONE cycle. After this runs:
    1. regenerate the dataset once
    2. re-run G3.5 + the independent audit once
    3. build model_gnn_attn.py
No iterating. If the new redundancy check (G3.5 check 8) flags anything further,
fold the fix into this script and re-run the whole pass -- do not start a second
audit cycle.

WHAT IT CHANGES
---------------
A. features_v2.py
   A1. EDGE_FEATURES: DROP 'snr'.  6 -> 5.
       The graph's `snr` attribute is `sinr_clean`, returned by _measured_link
       at interference_mw=0, n_contenders=1, shadowing_db=0. Both directions
       share dist_ij, so the 0.5*(s_ij+s_ji) average collapses to a
       deterministic, invertible function of `distance`. It carries NO load or
       interference information despite the name.
       NOTE -- this is a CHANGE OF PLAN from "replace snr with measured SINR".
       There is no measured SINR to substitute: the interference model is a
       Bernoulli lethal-interferer indicator (p_clear), not a continuous power
       sum, by design (reviewer item M-8). Any "measured SINR" would have to be
       synthesised from link_quality, creating a fresh redundancy. Dropping is
       the honest fix. The genuinely additive option -- exposing p_clear and
       p_coll separately, since the model can currently recover only their
       product -- is DEFERRED to M5 and recorded in the punch list.

   A2. QUERY_FEATURES: DROP 'hops_so_far'.  7 -> 6.
       ttl_left = max(TTL-hops,0)/TTL and hops_so_far = min(hops/TTL,1) sum to
       exactly 1.0 for every row (max observed hops 19 < TTL 20, so neither
       clip ever engages). Exact linear redundancy, Pearson r = -1.000.

   A3. QUERY_FEATURES rename + observability scoping:
       'n_inflight'             -> 'neigh_buffered_packets'
       'network_mean_occupancy' -> 'neigh_mean_occupancy'
       Both are now computed over the LOCAL_HORIZON-hop neighbourhood of the
       current node when nc['local_horizon'] is an int, and over the whole
       network when it is None. Renaming is not cosmetic: a column named
       `network_mean_occupancy` holding a 2-hop mean is the same naming trap as
       `snr`. horizon=None means "the neighbourhood is the whole network" and
       is the controller-assisted scoping choice.
       DEFAULT IS LOCAL_HORIZON = 2. Set to None in features_v2.py to declare
       the architecture controller-assisted instead. This is the one decision
       that must be signed off before regeneration.

   A4. degree normaliser moves into norm_constants as 'degree_ref'.
       Was G.degree(n)/max(N-1,1) with N read off the live graph and never
       persisted, so the "persist and reuse unchanged" guarantee did not cover
       it. Value is unchanged; it is now auditable.

   A5. FEATURE_SCHEMA_VERSION added, plus a structural self-test at import.

B. preflight_dataset_v2_check.py
   B1. NEW CHECK 8 -- pairwise feature redundancy, within each of the four
       blocks. Pearson |r| catches linear duplication (the ttl pair); Spearman
       |rho| catches monotone-nonlinear duplication (the snr/distance pair,
       whose linear r is only -0.93 and would have been waved through).
   B2. Manifest schema compatibility gate (see C).

C. audit_dataset_v2.py + preflight_dataset_v2_check.py
   C1. Both now assert the manifest's persisted feature lists match the live
       features_v2 module, and FAIL loudly on mismatch.
       Every lookup in both checkers is name-based
       (F.NODE_FEATURES.index('queue_occupancy'), enumerate(F.QUERY_FEATURES)),
       so nothing breaks by index arithmetic. What breaks is VERSION SKEW:
       running a checker against a dataset generated under a different
       features_v2.py resolves names against the new module while the stored
       array has the old layout. Silent misalignment, no exception, plausible
       numbers. generate_dataset_v2.py already persists all four lists
       (manifest keys node_features/edge_features/query_features/
       candidate_features) and no checker ever read them back -- the same
       persisted-but-unused shape as the norm_constants warning.

D. experiment_spbp_mechanism.py, experiment_queue_weight.py
   D1. Add --out and json.dump. These two scripts were never able to write
       results, unlike headroom / locality_cost / collision_model /
       calibration_sensitivity, which all have both. The Findings 2 and 4
       numbers therefore have no committed evidence.

E. generate_dataset_v2.py
   E1. Persist FEATURE_SCHEMA_VERSION and the observability scoping choice in
       manifest.json.

USAGE
-----
    python apply_pre_m4_fixes_v1.py --src src [--dry-run]

Every replacement is anchored and asserted. A non-matching anchor raises
PatchError and NOTHING is written -- the script stages all edits in memory and
commits only if every anchor matched.
"""

import os
import sys
import argparse

VERSION = "v1"


class PatchError(RuntimeError):
    pass


def sub(text, old, new, label, path):
    """Assertion-guarded replace. Anchor must match exactly once."""
    n = text.count(old)
    if n != 1:
        raise PatchError(
            f"[{path}] anchor for '{label}' matched {n} times, expected exactly 1.\n"
            f"  First 120 chars of anchor: {old[:120]!r}"
        )
    return text.replace(old, new, 1)


# ─────────────────────────────────────────────────────────────────────────────
# A. features_v2.py
# ─────────────────────────────────────────────────────────────────────────────

A_OLD_LISTS = """EDGE_FEATURES = [
    'distance',
    'link_quality',
    'packet_error_rate',
    'snr',
    'estimated_link_lifetime',
    'relative_velocity',
]

QUERY_FEATURES = [
    'ttl_left',
    'hops_so_far',
    'dist_to_dest',
    'current_queue_occupancy',
    'n_inflight',
    'network_mean_occupancy',
    'hop_distance_to_dst',
]"""

A_NEW_LISTS = '''EDGE_FEATURES = [
    'distance',
    'link_quality',
    'packet_error_rate',
    'estimated_link_lifetime',
    'relative_velocity',
]
# REMOVED FEATURE -- 'snr' sat between 'packet_error_rate' and
# 'estimated_link_lifetime'. The graph attribute it read is `sinr_clean`,
# returned by simulator_v2._measured_link from
# compute_link_features_v2(dist, interference_mw=0.0, n_contenders=1,
# shadowing_db=0.0). Both link directions pass the same dist_ij, so the
# 0.5*(s_ij+s_ji) average collapses to a deterministic, strictly monotone
# function of `distance`. Linear r is only about -0.93, which is why a
# correlation screen alone would have let it through -- but the map is
# invertible, so any nonlinear model recovers one column exactly from the
# other. It carried NO interference or load information despite its name.
# There is no measured SINR available to substitute: the interference model is
# a Bernoulli lethal-interferer indicator, not a continuous power sum (M-8).
# DEFERRED to M5: expose p_clear and p_coll as separate edge features. The
# model can currently recover only their product, since
# link_quality = lq_clean * p_clear * (1 - p_coll), and hidden-terminal loss
# and MAC contention are physically distinct channels.

QUERY_FEATURES = [
    'ttl_left',
    'dist_to_dest',
    'current_queue_occupancy',
    'neigh_buffered_packets',
    'neigh_mean_occupancy',
    'hop_distance_to_dst',
]
# REMOVED FEATURE -- 'hops_so_far' sat between 'ttl_left' and 'dist_to_dest'.
#   ttl_left    = max(TTL - hops, 0) / TTL
#   hops_so_far = min(hops / TTL, 1)
# These sum to exactly 1.0 on every row: max observed hops is 19 against
# TTL = 20, so neither clip ever engages. Exact LINEAR redundancy,
# Pearson r = -1.000. Same defect class as the queue_len removal below, and
# it would have split feature importance in the one milestone whose purpose is
# justifying the architecture.
#
# RENAMED -- 'n_inflight' -> 'neigh_buffered_packets',
#            'network_mean_occupancy' -> 'neigh_mean_occupancy'.
# Both are now computed over the LOCAL_HORIZON-hop neighbourhood of the
# current node (see LOCAL_HORIZON below). A column named
# `network_mean_occupancy` that holds a 2-hop mean is the same naming trap the
# `snr` removal above is about, so the names move with the semantics.'''

A_OLD_CONSTS = """MAX_QUEUE_REF = 50.0
INITIAL_ENERGY_REF = 100.0
TTL_REF = 20.0
SNR_REF = 40.0          # dB, generous ceiling for scaling
LIFETIME_REF = 60.0     # seconds, matches estimate_link_lifetime's max
HOP_CAP = 10.0          # hop distances clipped/normalised against this"""

A_NEW_CONSTS = '''MAX_QUEUE_REF = 50.0
INITIAL_ENERGY_REF = 100.0
TTL_REF = 20.0
LIFETIME_REF = 60.0     # seconds, matches estimate_link_lifetime's max
HOP_CAP = 10.0          # hop distances clipped/normalised against this

# ── OBSERVABILITY SCOPING DECISION ───────────────────────────────────────────
# Two query features aggregate load beyond the current node. A distributed
# router cannot observe network-wide state, and the M4 encoder's dense N x N
# attention is likewise centralised. This constant is where that scoping
# decision is made explicit and persisted, rather than left implicit.
#
#   LOCAL_HORIZON = k    -> both features are computed over the k-hop
#                           neighbourhood. Locally computable: k-hop neighbour
#                           state is what OLSR-family protocols already
#                           exchange via HELLO messages, so the assumption is
#                           standard rather than novel.
#   LOCAL_HORIZON = None -> whole-network aggregation. This is the
#                           CONTROLLER-ASSISTED scoping choice and must be
#                           stated as such in the paper.
#
# Default 2 matches the shallow end of the M4 depth sweep L in {0,1,2,3} and
# the spbp_k1..k4 horizon pattern already established in
# routing_teachers_v3_local.py. It is also consistent with the measured result
# that global BFS hop-distance is worth only +0.0005 PDR: there is little
# evidence global aggregation buys anything here, and a large deployability
# cost to claiming it.
LOCAL_HORIZON = 2

# Bumped whenever the four feature lists change. Persisted in manifest.json and
# asserted by both checkers, so a checker can never resolve feature names
# against a module that disagrees with the dataset it is reading.
FEATURE_SCHEMA_VERSION = 2


def _assert_schema_sane():
    """Structural self-test at import. Cheap, and it fails loudly.

    This does NOT replace G3.5 check 8 -- name uniqueness is a property of the
    lists, redundancy is a property of the DATA and can only be measured on a
    generated dataset.
    """
    for nm, lst in (('NODE', NODE_FEATURES), ('EDGE', EDGE_FEATURES),
                    ('QUERY', QUERY_FEATURES), ('CANDIDATE', CANDIDATE_FEATURES)):
        assert len(set(lst)) == len(lst), f"{nm}_FEATURES contains a duplicate name"
        assert all(isinstance(x, str) and x for x in lst), f"{nm}_FEATURES malformed"
    assert LOCAL_HORIZON is None or (isinstance(LOCAL_HORIZON, int)
                                     and LOCAL_HORIZON >= 1), \\
        "LOCAL_HORIZON must be None (controller-assisted) or an int >= 1"


_assert_schema_sane()'''

A_OLD_NORM = """    return {
        'area_x': float(cfg['area_x']),
        'area_y': float(cfg['area_y']),
        'z_min': float(cfg['z_min']),
        'z_max': float(cfg['z_max']),
        'speed_max': float(cfg.get('speed_max', 15.0)),
        'comm_range': float(cfg['comm_range']),
        'max_queue': MAX_QUEUE_REF,
        'initial_energy': INITIAL_ENERGY_REF,
        'ttl': TTL_REF,
        'snr_ref': SNR_REF,
        'lifetime_ref': LIFETIME_REF,
        'hop_cap': HOP_CAP,
    }"""

A_NEW_NORM = """    return {
        'area_x': float(cfg['area_x']),
        'area_y': float(cfg['area_y']),
        'z_min': float(cfg['z_min']),
        'z_max': float(cfg['z_max']),
        'speed_max': float(cfg.get('speed_max', 15.0)),
        'comm_range': float(cfg['comm_range']),
        'max_queue': MAX_QUEUE_REF,
        'initial_energy': INITIAL_ENERGY_REF,
        'ttl': TTL_REF,
        'lifetime_ref': LIFETIME_REF,
        'hop_cap': HOP_CAP,
        # Degree normaliser. Was max(N-1, 1) read off the live graph and never
        # persisted, so the persist-and-reuse-unchanged guarantee did not cover
        # it. Same value, now auditable. Note it makes normalised degree
        # conflate density with network size: degree 18 is 0.409 at N=45 but
        # would be 0.947 at N=20, and medium_slow (4.37/29 = 0.151) sits almost
        # on top of sparse_fast (2.51/19 = 0.132) despite a 1.7x density gap.
        # State that when reporting the generalisation result.
        'degree_ref': float(max(int(cfg['num_drones']) - 1, 1)),
        # None = whole-network aggregation (controller-assisted).
        'local_horizon': LOCAL_HORIZON,
        'schema_version': FEATURE_SCHEMA_VERSION,
    }"""

A_OLD_DEGREE = "            G.degree(n) / max(N - 1, 1),"
A_NEW_DEGREE = "            G.degree(n) / nc['degree_ref'],"

A_OLD_EF = """        ef[k] = (
            e.get('distance', 0.0) / nc['comm_range'],
            e.get('link_quality', 0.0),
            e.get('packet_error_rate', 0.0),
            np.clip(e.get('snr', 0.0) / nc['snr_ref'], -1.0, 2.0),
            min(e.get('estimated_link_lifetime', 0.0), nc['lifetime_ref']) / nc['lifetime_ref'],
            min(e.get('relative_velocity', 0.0) / max(2.0 * smax, 1e-6), 1.0),
        )"""

A_NEW_EF = """        ef[k] = (
            e.get('distance', 0.0) / nc['comm_range'],
            e.get('link_quality', 0.0),
            e.get('packet_error_rate', 0.0),
            min(e.get('estimated_link_lifetime', 0.0), nc['lifetime_ref']) / nc['lifetime_ref'],
            min(e.get('relative_velocity', 0.0) / max(2.0 * smax, 1e-6), 1.0),
        )"""

A_OLD_QF = """    h_cur = h_map.get(c, nc['hop_cap'])
    qf = np.array([
        max(ttl_ref - pkt.hops, 0.0) / ttl_ref,
        min(pkt.hops / ttl_ref, 1.0),
        min(dist_cd / max(diag, 1e-6), 1.0),
        G.nodes[c].get('queue_occupancy', 0.0),
        min(n_inflight / 100.0, 1.0),
        network_mean_occ,
        min(h_cur / nc['hop_cap'], 1.0),
    ], dtype=np.float32)"""

A_NEW_QF = """    h_cur = h_map.get(c, nc['hop_cap'])

    # Observability scoping (see LOCAL_HORIZON). With a horizon set, both load
    # aggregates are restricted to the k-hop neighbourhood of the CURRENT node,
    # so every query feature is locally computable by the forwarding node. With
    # horizon None the caller's whole-network values are used unchanged, which
    # is the controller-assisted variant.
    k = nc.get('local_horizon')
    if k is not None:
        nbrs = nx.single_source_shortest_path_length(G, c, cutoff=int(k))
        occs = [G.nodes[u].get('queue_occupancy', 0.0) for u in nbrs]
        if occs:
            # Local restriction of exactly the same two quantities: mean queue
            # occupancy, and the packet count those occupancies represent
            # (occupancy IS length/MAX_QUEUE, so x max_queue recovers packets).
            neigh_mean_occ = float(np.mean(occs))
            neigh_buffered = float(np.sum(occs)) * nc['max_queue']
        else:
            neigh_mean_occ, neigh_buffered = 0.0, 0.0
    else:
        neigh_mean_occ = float(network_mean_occ)
        neigh_buffered = float(n_inflight)

    qf = np.array([
        max(ttl_ref - pkt.hops, 0.0) / ttl_ref,
        min(dist_cd / max(diag, 1e-6), 1.0),
        G.nodes[c].get('queue_occupancy', 0.0),
        min(neigh_buffered / 100.0, 1.0),
        neigh_mean_occ,
        min(h_cur / nc['hop_cap'], 1.0),
    ], dtype=np.float32)"""


# ─────────────────────────────────────────────────────────────────────────────
# B/C. shared schema-compatibility helper, injected into features_v2
# ─────────────────────────────────────────────────────────────────────────────

A_OLD_HOPDIST = '''def hop_distances_to(G, dst):'''

A_NEW_HOPDIST = '''MANIFEST_FEATURE_KEYS = (
    ('node_features', 'NODE_FEATURES'),
    ('edge_features', 'EDGE_FEATURES'),
    ('query_features', 'QUERY_FEATURES'),
    ('candidate_features', 'CANDIDATE_FEATURES'),
)


def assert_manifest_compatible(man, context=''):
    """Fail loudly if a dataset's persisted feature lists disagree with THIS module.

    Every feature lookup in both checkers is name-based
    (NODE_FEATURES.index('queue_occupancy'), enumerate(QUERY_FEATURES)), so
    nothing breaks by index arithmetic. What breaks is VERSION SKEW: a checker
    run against a dataset generated under a different features_v2.py resolves
    names against the new module while the stored array still has the old
    column layout. That misaligns silently -- no exception, plausible numbers,
    wrong conclusions. Exactly the failure mode the norm_constants warning
    describes.

    generate_dataset_v2.py has always persisted these four lists. Nothing ever
    read them back. This makes the manifest a real compatibility boundary
    rather than a documented one.

    Returns a list of human-readable mismatch strings; empty means compatible.
    """
    problems = []
    live_ver = FEATURE_SCHEMA_VERSION
    man_ver = man.get('feature_schema_version')
    if man_ver is None:
        problems.append(
            'manifest has no feature_schema_version: it predates the schema '
            'boundary and CANNOT be validated against this module. Regenerate.')
    elif man_ver != live_ver:
        problems.append(
            f'feature_schema_version mismatch: manifest={man_ver} '
            f'module={live_ver}')

    for man_key, mod_name in MANIFEST_FEATURE_KEYS:
        stored = man.get(man_key)
        live = globals()[mod_name]
        if stored is None:
            problems.append(f'manifest is missing {man_key!r}')
            continue
        if list(stored) != list(live):
            problems.append(
                f'{man_key}: manifest={list(stored)} != module={list(live)}')

    man_h = man.get('local_horizon', '<absent>')
    if man_h != LOCAL_HORIZON:
        problems.append(
            f'local_horizon mismatch: manifest={man_h} module={LOCAL_HORIZON}. '
            f'The observability scoping of the dataset differs from the code '
            f'about to read it.')

    if problems and context:
        problems = [f'({context}) {p}' for p in problems]
    return problems


def hop_distances_to(G, dst):'''


# ─────────────────────────────────────────────────────────────────────────────
# B. preflight_dataset_v2_check.py
# ─────────────────────────────────────────────────────────────────────────────

B_OLD_DOC = """  7. Reproducibility       — regenerating one episode with the same seed yields
                             an identical decision count and identical labels."""

B_NEW_DOC = """  7. Reproducibility       — regenerating one episode with the same seed yields
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
                             of them meaningless rather than merely wrong."""

B_OLD_CONST = """EPS_TOL = 0.02          # allowed deviation of measured epsilon from configured
FALLBACK_MAX = 0.05     # max acceptable label_fallback rate"""

B_NEW_CONST = '''EPS_TOL = 0.02          # allowed deviation of measured epsilon from configured
FALLBACK_MAX = 0.05     # max acceptable label_fallback rate

# Check 8 thresholds. Deliberately not 1.0: float32 storage against float64
# scoring puts a quantisation floor under any exact identity, the same reason
# the independent audit's label re-derivation threshold is 0.99.
PEARSON_MAX = 0.98      # linear duplication
SPEARMAN_MAX = 0.995    # monotone (possibly nonlinear) duplication
REDUNDANCY_SAMPLE = 200000


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
                offenders.append((block_name, names[i], names[j], p, s))
                kind = ('linear' if p > PEARSON_MAX else
                        'monotone (nonlinear -- Pearson alone would MISS this)')
                lines.append(f'    ** {block_name}: {names[i]} <-> {names[j]}  '
                             f'|r|={p:.4f} |rho|={s:.4f}  [{kind}]')
    return offenders, lines'''

B_OLD_LOAD = """    dec, frm, man = load(args.data)
    n = len(dec['label'])"""

B_NEW_LOAD = """    dec, frm, man = load(args.data)

    # ---- 0. schema compatibility (runs FIRST, aborts on failure) ----
    skew = F.assert_manifest_compatible(man, context='G3.5')
    if skew:
        print("\\n" + "=" * 78)
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
        print("=" * 78 + "\\n")
        return 1

    n = len(dec['label'])"""

B_OLD_C7 = """    c7 = len(set(man['seeds'])) == len(man['seeds'])"""

B_NEW_C7 = """    c7 = len(set(man['seeds'])) == len(man['seeds'])

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
    c8 = not red_offenders"""

B_OLD_DEADPRINT = """    if dead_node or dead_query:
        print(f"    ** DEAD (zero-variance) FEATURES: node={dead_node} query={dead_query}")"""

B_NEW_DEADPRINT = """    if dead_node or dead_query:
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
        print("      no redundant pairs inside any block")"""

B_OLD_CHECKS = """        ("7. Seed list well-formed (no duplicates)", c7,
         f"{len(man['seeds'])} unique seeds"),
    ]"""

B_NEW_CHECKS = """        ("7. Seed list well-formed (no duplicates)", c7,
         f"{len(man['seeds'])} unique seeds"),
        ("8. No redundant feature pairs within a block", c8,
         (f"{len(red_offenders)} offending pair(s)" if red_offenders
          else f"none (|r|<={PEARSON_MAX}, |rho|<={SPEARMAN_MAX})")),
    ]"""


# ─────────────────────────────────────────────────────────────────────────────
# C. audit_dataset_v2.py
# ─────────────────────────────────────────────────────────────────────────────

C_OLD_MAN = """    with open(os.path.join(args.data, 'manifest.json')) as f:
        man = json.load(f)"""

C_NEW_MAN = """    with open(os.path.join(args.data, 'manifest.json')) as f:
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
        print("\\n" + "=" * 78)
        print("  AUDIT ABORTED — DATASET/MODULE SCHEMA SKEW")
        print("=" * 78)
        for _p in _skew:
            print(f"    ** {_p}")
        print()
        print("    Refusing to audit: name-based column lookups would resolve")
        print("    against a layout the dataset does not have.")
        print("=" * 78 + "\\n")
        return 1"""


# ─────────────────────────────────────────────────────────────────────────────
# D. experiment JSON output
# ─────────────────────────────────────────────────────────────────────────────

D_SPBP_OLD_ARGS = """    ap.add_argument('--part', choices=['A', 'B', 'C', 'all'], default='all')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()"""

D_SPBP_NEW_ARGS = """    ap.add_argument('--part', choices=['A', 'B', 'C', 'all'], default='all')
    ap.add_argument('--quick', action='store_true')
    # Findings 2 and 4 rest on this script's numbers and it could not write
    # them. headroom / locality_cost / collision_model /
    # calibration_sensitivity all take --out and json.dump; these two never
    # did, so their results existed only in a terminal.
    ap.add_argument('--out', default='results/spbp_mechanism.json',
                    help='write results here; "" disables')
    args = ap.parse_args()"""

D_SPBP_OLD_RUN = """    if args.part in ('A', 'all'):
        part_a(args.seeds)
    if args.part in ('B', 'all'):
        part_b(args.seeds)
    if args.part in ('C', 'all'):
        part_c(args.seeds, args.rates, args.max_workers, args.quick)

    print("\\n" + "=" * 78 + "\\n")
    return 0"""

D_SPBP_NEW_RUN = """    out = {'seeds': list(args.seeds), 'rates': list(args.rates),
           'part': args.part, 'quick': bool(args.quick)}
    if args.part in ('A', 'all'):
        out['part_a_geo_hop'] = part_a(args.seeds)
    if args.part in ('B', 'all'):
        out['part_b'] = part_b(args.seeds)
    if args.part in ('C', 'all'):
        means = part_c(args.seeds, args.rates, args.max_workers, args.quick)
        out['part_c_means'] = means
        full = means.get('spbp_ab_full')
        if full is not None:
            out['part_c_cost_of_removal'] = {
                k: (full - means[k]) for k in
                ('spbp_ab_noqueue', 'spbp_ab_candqueue', 'spbp_ab_additive')
                if k in means}

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(_jsonable(out), f, indent=2)
        print(f"  wrote {args.out}")

    print("\\n" + "=" * 78 + "\\n")
    return 0"""

D_QW_OLD_ARGS = """    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--quick', action='store_true')"""

D_QW_NEW_ARGS = """    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--quick', action='store_true')
    # See the note in experiment_spbp_mechanism.main: this script could not
    # write its results either, and Finding 4's refutation depends on them.
    ap.add_argument('--out', default='results/queue_weight.json',
                    help='write results here; "" disables')"""

D_QW_OLD_END = """    print("=" * 78 + "\\n")
    return 0"""

D_QW_NEW_END = """    if args.out:
        out = {
            'seeds': list(args.seeds), 'rates': list(args.rates),
            'da_weights': list(DA_WEIGHTS), 'means': _jsonable(means),
            'spbp_reference': float(sp_ref), 'dagpsr_reference': float(da_ref),
            'gap': float(gap),
            'up_fraction_closed': float(up_closed),
            'down_fraction_given': float(down_given),
            'best_da_weight': float(best_w), 'best_da_pdr': float(best_da),
            'verdict': ('scale_not_structure' if (up_closed > 0.7 and down_given > 0.7)
                        else 'one_direction_only' if (up_closed > 0.7 or down_given > 0.7)
                        else 'refuted_structural'),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {args.out}")
    print("=" * 78 + "\\n")
    return 0"""

JSONABLE = '''

def _jsonable(o):
    """numpy scalars/arrays -> plain Python, recursively. json.dump chokes on
    np.float32 and on tuple dict keys, both of which the oracle tables use."""
    import numpy as _np
    if isinstance(o, dict):
        return {(str(k) if not isinstance(k, str) else k): _jsonable(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, _np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, float) and (o != o):
        return None
    return o
'''


# ─────────────────────────────────────────────────────────────────────────────
# E. generate_dataset_v2.py — persist the schema + scoping decision
# ─────────────────────────────────────────────────────────────────────────────

E_OLD = """        'node_features': F.NODE_FEATURES,
        'edge_features': F.EDGE_FEATURES,
        'query_features': F.QUERY_FEATURES,
        'candidate_features': F.CANDIDATE_FEATURES,"""

E_NEW = """        'node_features': F.NODE_FEATURES,
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
        'local_horizon': F.LOCAL_HORIZON,"""


PATCHES = {
    'features_v2.py': [
        (A_OLD_LISTS, A_NEW_LISTS, 'A1/A2/A3 feature lists'),
        (A_OLD_CONSTS, A_NEW_CONSTS, 'A3/A5 LOCAL_HORIZON + schema version'),
        (A_OLD_NORM, A_NEW_NORM, 'A4 norm_constants'),
        (A_OLD_DEGREE, A_NEW_DEGREE, 'A4 degree normaliser'),
        (A_OLD_EF, A_NEW_EF, 'A1 edge feature tuple'),
        (A_OLD_QF, A_NEW_QF, 'A2/A3 query feature tuple'),
        (A_OLD_HOPDIST, A_NEW_HOPDIST, 'C1 assert_manifest_compatible'),
    ],
    'preflight_dataset_v2_check.py': [
        (B_OLD_DOC, B_NEW_DOC, 'B docstring'),
        (B_OLD_CONST, B_NEW_CONST, 'B1 redundancy helpers'),
        (B_OLD_LOAD, B_NEW_LOAD, 'B2 schema gate'),
        (B_OLD_C7, B_NEW_C7, 'B1 check 8'),
        (B_OLD_DEADPRINT, B_NEW_DEADPRINT, 'B1 diagnostics'),
        (B_OLD_CHECKS, B_NEW_CHECKS, 'B1 verdict row'),
    ],
    'audit_dataset_v2.py': [
        (C_OLD_MAN, C_NEW_MAN, 'C1 schema gate'),
    ],
    'experiment_spbp_mechanism.py': [
        (D_SPBP_OLD_ARGS, D_SPBP_NEW_ARGS, 'D1 --out'),
        (D_SPBP_OLD_RUN, D_SPBP_NEW_RUN, 'D1 json.dump'),
        ('\n\ndef main():', JSONABLE + '\ndef main():', 'D1 _jsonable helper'),
    ],
    'experiment_queue_weight.py': [
        (D_QW_OLD_ARGS, D_QW_NEW_ARGS, 'D1 --out'),
        (D_QW_OLD_END, D_QW_NEW_END, 'D1 json.dump'),
        ('\n\ndef main():', JSONABLE + '\ndef main():', 'D1 _jsonable helper'),
    ],
    'generate_dataset_v2.py': [
        (E_OLD, E_NEW, 'E1 persist schema + scoping'),
    ],
}

# Files needing `import json` / `import os` that may not already have them.
NEEDS_IMPORTS = {
    'experiment_spbp_mechanism.py': ('os', 'json'),
    'experiment_queue_weight.py': ('os', 'json'),
}

ALREADY_APPLIED_MARKER = 'FEATURE_SCHEMA_VERSION'


def ensure_imports(text, mods, path):
    for m in mods:
        if f'\nimport {m}' in text or f'import {m},' in text or f', {m}' in text.split('\n\n')[0]:
            continue
        anchor = 'import numpy as np'
        if anchor not in text:
            raise PatchError(f'[{path}] cannot find import anchor to add {m!r}')
        text = text.replace(anchor, f'import {m}\n{anchor}', 1)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    fv2 = os.path.join(args.src, 'features_v2.py')
    if not os.path.isfile(fv2):
        print(f"ERROR: {fv2} not found. Run from the repo root, or pass --src.")
        return 2
    with open(fv2, encoding='utf-8') as f:
        if ALREADY_APPLIED_MARKER in f.read():
            print("ERROR: features_v2.py already contains "
                  f"{ALREADY_APPLIED_MARKER!r}.\n"
                  "  This patch has already been applied. Re-applying would "
                  "corrupt the file.\n"
                  "  If you need to re-run it, restore src/ from git first.")
            return 2

    staged = {}
    print(f"\n{'=' * 78}\n  PRE-M4 FIX PASS {VERSION} — assertion-guarded\n{'=' * 78}")
    for fname, edits in PATCHES.items():
        path = os.path.join(args.src, fname)
        if not os.path.isfile(path):
            raise PatchError(f'missing file: {path}')
        with open(path, encoding='utf-8') as f:
            text = f.read()
        for old, new, label in edits:
            text = sub(text, old, new, label, fname)
            print(f"    [ok] {fname:<34} {label}")
        if fname in NEEDS_IMPORTS:
            text = ensure_imports(text, NEEDS_IMPORTS[fname], fname)
            print(f"    [ok] {fname:<34} imports")
        staged[path] = text

    if args.dry_run:
        print(f"\n  DRY RUN — {len(staged)} file(s) would be written, nothing changed.")
        return 0

    for path, text in staged.items():
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
    print(f"\n  WROTE {len(staged)} file(s).")
    print("\n  NEXT (in this order, once each):")
    print("    1. python src/features_v2.py          # import self-test")
    print("    2. regenerate:  python src/generate_dataset_v2.py ...")
    print("    3. python src/preflight_dataset_v2_check.py --data data/phaseB")
    print("    4. python src/audit_dataset_v2.py --data data/phaseB --sample 20000")
    print("  The OLD dataset will now correctly ABORT both checkers with a")
    print("  schema-skew message. That is the boundary working, not a failure.")
    print("=" * 78 + "\n")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except PatchError as e:
        print(f"\nPATCH FAILED — nothing was written.\n{e}\n")
        sys.exit(1)
