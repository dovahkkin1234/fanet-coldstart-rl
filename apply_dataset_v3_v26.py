"""apply_dataset_v3_v26.py -- code patch for Dataset V3 (docs/DATASET_V3_SPEC.md).

    python apply_dataset_v3_v26.py --src src --dry-run
    python apply_dataset_v3_v26.py --src src
    python verify_dataset_v3_v26.py --src src

Assertion-guarded str.replace, per the repo contract (docs/CLAUDE.md): every anchor
must match its target exactly the stated number of times, all edits are staged in
memory, and NOTHING is written unless every anchor in every file matches. A file
whose guard string is already present is reported as already applied and left
alone (idempotent). New files (generate_dataset_v3.py, teacher_pickers_v3.py, the
v3 checks, exporter, gate, grid verifier) are copied in separately -- this script
only edits existing files.

WHAT IT CHANGES (each item is traced in the spec)
  config_v2.py        + OPERATING_POINT, DATASET_GRID (band + low anchor), splits,
                        rate_role / rate_index / load_bucket_rel, dataset_episode_config.
                        BASE / RATES / SCENARIOS untouched (still the parity reference;
                        apply_v8b_operating_point_STAGED.py still matches afterwards).
  features_v2.py      FEATURE_SCHEMA_VERSION 5 -> 6. Query column 2 renamed
                        current_queue_occupancy -> own_queue_live and read LIVE at
                        decision time (caller must pass it; v6 refuses a missing value).
                        BUFFERED_REF re-sized for v6 from post-v12 data (v5 keeps 500).
                        norm_constants(cfg, schema_version=None); v5 lists kept in
                        LEGACY_FEATURE_LISTS; assert_manifest_compatible(...,
                        accept_legacy=False) reads a v5 dataset only when asked.
  train_supervised_v2 PhaseB(accept_legacy=False, use_weights=False); optional
                        per-row loss weights (context multiplicity); the unweighted
                        path is byte-identical to before.
  rollout_eval_v2.py  ModelActorSimulator(schema_version=...): v5 snapshot / v6 live
                        own-queue semantics follow the model's training data; the
                        legacy G4 script reads its (v5) dataset with accept_legacy.
  generate_dataset_v2 pinned to schema v5 (legacy), main() needs --legacy_v5, the
                        false 'SP-BP wins all 12 cells' comment corrected.
  audit_dataset_v2 / preflight_dataset_v2_check
                      read legacy v5 manifests explicitly; column names come from
                        the manifest; the wrong dequeue explanation corrected (R-24).
  find_congestion_band_convergecast.py
                      --z_min/--z_max (default 100/300, like find_usable_band.py)
                        and recorded in run_params -- sink_50's band was measured at
                        BASE altitude 50-150 m because this flag did not exist.
"""
import argparse
import io
import os
import sys

BUFFERED_REF_V6 = None   # set below, after the measurement block

# ─────────────────────────────────────────────────────────────────────────────
# config_v2.py  (append-only, so v8b's anchors are untouched)
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_GUARD = "OPERATING_POINT = dict("
CONFIG_BLOCK = r'''

# ── v26: DATASET V3 operating point and grid ───────────────────────────────
# BASE and RATES above stay the 40 s / 50-150 m PARITY REFERENCE until v8b
# (decision D-12). The dataset is generated at the measured operating point
# EXPLICITLY, through dataset_episode_config(), so the dataset never depends on
# BASE having moved and moving BASE never silently changes the dataset.
# docs/DATASET_V3_SPEC.md is the specification for everything below.
OPERATING_POINT = dict(z_min=100, z_max=300, duration=1000.0, drain_time=10.0,
                       interference_on=True, initial_energy=8000.0)

# Per-scenario grid (decision 2026-10-02: usable band + one low-load anchor).
#   band   = usable rates at 1000 s / battery 8000 / 100-300 m under the band
#            search's reference actor spbp_ab_noqueue (results/band_*_1000s.json,
#            criteria elasticity 0.05-0.85, q_ovf >= 0.02, energy <= 0.05).
#            The band is a property of the LOAD, measured with a fixed reference
#            policy; it is deliberately NOT re-defined under the dataset's own
#            behaviour policy (that would let a better policy move the band).
#   anchor = the lowest rate of that scenario's band sweep: below the band on
#            purpose, for the low-vs-high load contrast.
# Buckets are scenario-RELATIVE: anchor -> 'low', top band rate -> 'high',
# other band rates -> 'medium'. Never compare bucket labels with pre-v26 results
# (those used absolute thresholds 0.5 / 2.0 on the old grid).
# sink_50's band file was measured at 50-150 m (BASE); verify_dataset_grid_v3
# marks it UNVERIFIED until it is re-measured at 100-300 m, and the generator
# refuses unverified cells unless explicitly allowed.
DATASET_GRID = {
    'dense_slow':  {'suite': 'A', 'anchor': [40.0], 'band': [60.0, 80.0, 100.0]},
    'very_dense':  {'suite': 'A', 'anchor': [40.0], 'band': [60.0, 80.0]},
    'medium_slow': {'suite': 'A', 'anchor': [30.0], 'band': [40.0, 60.0, 80.0]},
    'sparse_fast': {'suite': 'A', 'anchor': [40.0], 'band': [80.0, 100.0]},
    'sink_50':     {'suite': 'C', 'anchor': [20.0], 'band': [30.0]},
}
DATASET_SEEDS = list(range(101, 151))
DATASET_SPLITS = {'train': (101, 135), 'val': (136, 142), 'test': (143, 150)}
GENERALISATION_SCENARIO = 'medium_slow'


def dataset_rates(scenario):
    g = DATASET_GRID[scenario]
    return sorted(g['anchor'] + g['band'])


def dataset_cells(scenarios=None):
    """[(scenario, rate)] in a fixed, documented order."""
    out = []
    for sc in DATASET_GRID:
        if scenarios is None or sc in scenarios:
            out += [(sc, r) for r in dataset_rates(sc)]
    return out


def rate_role(scenario, rate):
    g = DATASET_GRID[scenario]
    if float(rate) in g['anchor']:
        return 'anchor'
    if float(rate) in g['band']:
        return 'band'
    raise KeyError(f'{scenario}@{rate} is not in DATASET_GRID')


def rate_index(scenario, rate):
    return dataset_rates(scenario).index(float(rate))


def load_bucket_rel(scenario, rate):
    """Scenario-relative load bucket: 'low' / 'medium' / 'high'."""
    if rate_role(scenario, rate) == 'anchor':
        return 'low'
    return 'high' if float(rate) == max(DATASET_GRID[scenario]['band']) else 'medium'


def split_of_seed(seed):
    for name, (lo, hi) in DATASET_SPLITS.items():
        if lo <= int(seed) <= hi:
            return name
    raise KeyError(f'seed {seed} belongs to no dataset split {DATASET_SPLITS}')


def dataset_scenario_cfg(scenario):
    suite = 'C' if DATASET_GRID[scenario]['suite'] == 'C' else 'A'
    return get_suite(suite)[scenario]


def dataset_episode_config(scenario, rate, seed, **over):
    """Full episode config at the DATASET operating point (not BASE)."""
    return {**BASE, **OPERATING_POINT, **dataset_scenario_cfg(scenario),
            'packet_rate': float(rate), 'seed': int(seed), **over}


def dataset_provenance():
    return {'operating_point': dict(OPERATING_POINT),
            'grid': {k: dict(v) for k, v in DATASET_GRID.items()},
            'seeds': [DATASET_SEEDS[0], DATASET_SEEDS[-1]],
            'splits': {k: list(v) for k, v in DATASET_SPLITS.items()},
            'generalisation_scenario': GENERALISATION_SCENARIO,
            'parity_reference_base': dict(BASE),
            'config_module': 'config_v2'}
'''


def features_edits():
    return [
        ("QUERY_FEATURES = [\n"
         "    'ttl_left',\n"
         "    'dist_to_dest',\n"
         "    'current_queue_occupancy',\n"
         "    'neigh_buffered_packets',\n",
         "QUERY_FEATURES = [\n"
         "    'ttl_left',\n"
         "    'dist_to_dest',\n"
         "    'own_queue_live',          # v6 -- v5 called it 'current_queue_occupancy'\n"
         "    'neigh_buffered_packets',\n", 1),

        ("# It adds NO INFORMATION the model lacked. SP-BP excludes unreachable\n",
         "# v6 NOTE: the paragraph below was written for SP-BP labels. Dataset V3\n"
         "# labels with da_gpsr, which is geographic and DOES pick unreachable\n"
         "# candidates (measured: every decision whose destination is unreachable --\n"
         "# 10.6% in medium_slow@60, 70% in sparse_fast@100). The column's purpose,\n"
         "# ablatability, is unchanged.\n"
         "#\n"
         "# It adds NO INFORMATION the model lacked. SP-BP excludes unreachable\n", 1),

        ("BUFFERED_REF = 500.0\n",
         "BUFFERED_REF = 500.0\n"
         "# v6 (Dataset V3): the 500 above was sized on PRE-v12 data, in which\n"
         "# delivered packets were never released from destination queues (88.4% of\n"
         "# recorded drops were phantom), so the 2-hop buffered counts were inflated.\n"
         "# Post-fix, at the new operating point (1000 s, 100-300 m, battery 8000,\n"
         "# da_gpsr actor, top band rate of every scenario) the raw maximum is\n"
         "# {BUFFERED_MEASURED} -- 500 used at most {BUFFERED_USE} of the\n"
         "# feature's range. Sized like before (no clipping at the observed maximum),\n"
         "# with ~40% headroom because 800 episodes will reach further into the tail than\n"
         "# five; G3.5 v3's saturation diagnostic re-checks it on\n"
         "# the real dataset. v5 data keeps 500 (norm_constants(..., schema_version=5)).\n"
         "BUFFERED_REF_V6 = {BUFFERED_REF_V6}\n", 1),

        ("FEATURE_SCHEMA_VERSION = 5\n",
         "FEATURE_SCHEMA_VERSION = 6\n"
         "# v6 (Dataset V3, decision 2026-10-02). Two changes, both semantic:\n"
         "#   * query column 2 is the deciding node's OWN queue read LIVE at decision\n"
         "#     time (excluding the packet being forwarded) and renamed own_queue_live.\n"
         "#     v5's current_queue_occupancy was the frame-start snapshot: a byte-for-\n"
         "#     byte copy of node_feat[current].queue_occupancy, non-empty in 6-36% of\n"
         "#     decisions against 68-89% live. Neighbour and 2-hop queue signals stay\n"
         "#     the frame-start snapshot -- what the da_gpsr label reads, and what a\n"
         "#     real node learns about its neighbours by beacon.\n"
         "#   * neigh_buffered_packets is normalised by BUFFERED_REF_V6.\n"
         "# The name changes with the semantics so a v5 model can never be fed v6\n"
         "# columns silently; assert_manifest_compatible refuses v5 data unless the\n"
         "# reader passes accept_legacy=True and then uses v5 semantics itself.\n"
         "QUERY_FEATURES_V5 = ['ttl_left', 'dist_to_dest', 'current_queue_occupancy',\n"
         "                     'neigh_buffered_packets', 'neigh_mean_occupancy',\n"
         "                     'hop_distance_to_dst']\n"
         "LEGACY_FEATURE_LISTS = {5: {'node_features': list(NODE_FEATURES),\n"
         "                            'edge_features': list(EDGE_FEATURES),\n"
         "                            'query_features': list(QUERY_FEATURES_V5),\n"
         "                            'candidate_features': list(CANDIDATE_FEATURES)}}\n"
         "SUPPORTED_SCHEMA_VERSIONS = (5, 6)\n", 1),

        ("def norm_constants(cfg):\n"
         "    \"\"\"Per-scenario normalisation constants. Persist these with the dataset.\"\"\"\n"
         "    return {\n",
         "def norm_constants(cfg, schema_version=None):\n"
         "    \"\"\"Per-scenario normalisation constants. Persist these with the dataset.\n"
         "\n"
         "    schema_version selects the feature SEMANTICS (v26): None = this module's\n"
         "    FEATURE_SCHEMA_VERSION; 5 = legacy (snapshot own queue, BUFFERED_REF 500).\n"
         "    The returned 'schema_version' is what extract_decision obeys.\"\"\"\n"
         "    ver = int(FEATURE_SCHEMA_VERSION if schema_version is None else schema_version)\n"
         "    if ver not in SUPPORTED_SCHEMA_VERSIONS:\n"
         "        raise ValueError(f'feature schema v{ver} not supported: {SUPPORTED_SCHEMA_VERSIONS}')\n"
         "    return {\n", 1),

        ("        'buffered_ref': BUFFERED_REF,\n",
         "        'buffered_ref': BUFFERED_REF if ver < 6 else BUFFERED_REF_V6,\n", 1),

        ("        'local_horizon': LOCAL_HORIZON,\n"
         "        'schema_version': FEATURE_SCHEMA_VERSION,\n"
         "    }\n",
         "        'local_horizon': LOCAL_HORIZON,\n"
         "        'schema_version': ver,\n"
         "    }\n", 1),

        ("def assert_manifest_compatible(man, context=''):\n",
         "def assert_manifest_compatible(man, context='', accept_legacy=False):\n", 1),

        ("    problems = []\n"
         "    live_ver = FEATURE_SCHEMA_VERSION\n"
         "    man_ver = man.get('feature_schema_version')\n"
         "    if man_ver is None:\n",
         "    problems = []\n"
         "    live_ver = FEATURE_SCHEMA_VERSION\n"
         "    man_ver = man.get('feature_schema_version')\n"
         "    # v26: a LEGACY (v5) dataset is checked against the v5 lists, and only\n"
         "    # when the caller asks; the caller must then use v5 semantics itself.\n"
         "    ref_lists = None\n"
         "    if (man_ver is not None and man_ver != live_ver and accept_legacy\n"
         "            and man_ver in LEGACY_FEATURE_LISTS):\n"
         "        ref_lists = LEGACY_FEATURE_LISTS[man_ver]\n"
         "        man_ver = live_ver          # version accepted deliberately\n"
         "    if man_ver is None:\n", 1),

        ("        live = globals()[mod_name]\n",
         "        live = (ref_lists[man_key] if ref_lists is not None\n"
         "                else globals()[mod_name])\n", 1),

        ("def extract_decision(G, pkt, candidates, nc, h_map, n_inflight,\n"
         "                     network_mean_occ, ttl_const=None):\n",
         "def extract_decision(G, pkt, candidates, nc, h_map, n_inflight,\n"
         "                     network_mean_occ, ttl_const=None, own_queue_occ=None):\n", 1),

        ("    order — see generate_dataset_v2.canonical_candidates.\n"
         "    \"\"\"\n",
         "    order — see generate_dataset_v2.canonical_candidates.\n"
         "\n"
         "    own_queue_occ (v6): the deciding node's queue occupancy read LIVE at\n"
         "    decision time, excluding the packet being forwarded. REQUIRED when\n"
         "    nc['schema_version'] >= 6 -- a missing value raises rather than silently\n"
         "    falling back to the frame-start snapshot. Ignored for v5 semantics.\n"
         "    \"\"\"\n", 1),

        ("    qf = np.array([\n"
         "        max(ttl_ref - pkt.hops, 0.0) / ttl_ref,\n"
         "        min(dist_cd / max(diag, 1e-6), 1.0),\n"
         "        G.nodes[c].get('queue_occupancy', 0.0),\n",
         "    if int(nc.get('schema_version', 5)) >= 6:\n"
         "        if own_queue_occ is None:\n"
         "            raise ValueError(\n"
         "                'feature schema v6: own_queue_live must be read LIVE at decision '\n"
         "                'time and passed as own_queue_occ (e.g. sim.queues[c].occupancy); '\n"
         "                'refusing to fall back to the frame-start snapshot silently')\n"
         "        own_q = float(own_queue_occ)\n"
         "    else:\n"
         "        own_q = G.nodes[c].get('queue_occupancy', 0.0)\n"
         "    qf = np.array([\n"
         "        max(ttl_ref - pkt.hops, 0.0) / ttl_ref,\n"
         "        min(dist_cd / max(diag, 1e-6), 1.0),\n"
         "        own_q,\n", 1),
    ]


TRAIN_EDITS = [
    ("    def __init__(self, data_dir, mask=None):\n",
     "    def __init__(self, data_dir, mask=None, accept_legacy=False, use_weights=False):\n", 1),
    ("        skew = F.assert_manifest_compatible(self.man, context='M4 training')\n",
     "        skew = F.assert_manifest_compatible(self.man, context='M4 training',\n"
     "                                            accept_legacy=accept_legacy)\n"
     "        # v26: the dataset's OWN schema version decides the feature semantics a\n"
     "        # model trained on it must be rolled out with (v5 snapshot, v6 live).\n"
     "        self.schema_version = int(self.man.get('feature_schema_version', 5))\n"
     "        self.use_weights = bool(use_weights)\n"
     "        if self.use_weights and 'weight' not in self.dec:\n"
     "            raise SystemExit('use_weights=True but decisions.npz has no weight column '\n"
     "                             '(export_phaseb_v3.py --weighting multiplicity writes it)')\n", 1),
    ("            label=torch.from_numpy(d['label'][ids].astype(np.int64)).to(device),\n"
     "            ids=ids)\n",
     "            label=torch.from_numpy(d['label'][ids].astype(np.int64)).to(device),\n"
     "            weight=(torch.from_numpy(d['weight'][ids].astype(np.float32)).to(device)\n"
     "                    if getattr(self, 'use_weights', False) else None),\n"
     "            ids=ids)\n", 1),
    ("    out['n'] = int(len(correct))\n    return out\n",
     "    out['n'] = int(len(correct))\n"
     "    # v26: frequency-weighted accuracy when the rows are de-duplicated contexts\n"
     "    # carrying their multiplicity (Dataset V3) -- comparable to per-decision\n"
     "    # accuracy on a raw dataset. Reported only; early stopping is unchanged.\n"
     "    if 'weight' in ds.dec:\n"
     "        w = ds.dec['weight'][ids].astype(np.float64)\n"
     "        out['accuracy_raw_w'] = float((correct * w).sum() / max(w.sum(), 1e-12))\n"
     "        out['accuracy_contested_w'] = (float((correct[con] * w[con]).sum()\n"
     "                                             / max(w[con].sum(), 1e-12))\n"
     "                                       if con.any() else float('nan'))\n"
     "    return out\n", 1),
    ("    lossf = nn.CrossEntropyLoss()\n",
     "    lossf = nn.CrossEntropyLoss()\n"
     "    lossf_w = nn.CrossEntropyLoss(reduction='none')   # v26 weighted path only\n", 1),
    ("            loss = lossf(logits, bt['label'])\n",
     "            if bt.get('weight') is None:\n"
     "                loss = lossf(logits, bt['label'])     # unchanged, byte-identical\n"
     "            else:\n"
     "                w = bt['weight']\n"
     "                loss = (lossf_w(logits, bt['label']) * w).sum() / w.sum().clamp_min(1e-12)\n", 1),
]

ROLLOUT_EDITS = [
    ("    def __init__(self, config, model, device, mask=None):\n"
     "        super().__init__(config)\n"
     "        self.model = model\n"
     "        self.device = device\n"
     "        self.nc = F.norm_constants(config)\n",
     "    def __init__(self, config, model, device, mask=None, schema_version=None):\n"
     "        super().__init__(config)\n"
     "        self.model = model\n"
     "        self.device = device\n"
     "        # v26: feature SEMANTICS follow the data the model was trained on --\n"
     "        # v5 = frame-start snapshot own queue, v6 = live own queue.\n"
     "        self.schema_version = int(F.FEATURE_SCHEMA_VERSION if schema_version is None\n"
     "                                  else schema_version)\n"
     "        self.nc = F.norm_constants(config, schema_version=self.schema_version)\n", 1),
    ("        qf, cf = F.extract_decision(G, pkt, cands, self.nc, h_map, 0.0, 0.0,\n"
     "                                    ttl_const=TTL)\n"
     "        # Query and candidate blocks.",
     "        qf, cf = F.extract_decision(\n"
     "            G, pkt, cands, self.nc, h_map, 0.0, 0.0, ttl_const=TTL,\n"
     "            own_queue_occ=(self.queues[pkt.current].occupancy\n"
     "                           if self.schema_version >= 6 else None))\n"
     "        # Query and candidate blocks.", 1),
    ("def assert_mask_applied(mask_names, device='cpu'):\n",
     "def assert_mask_applied(mask_names, device='cpu', schema_version=None):\n", 1),
    ("                              model=None, device=device, mask=mask_names)\n",
     "                              model=None, device=device, mask=mask_names,\n"
     "                              schema_version=schema_version)\n", 1),
    ("        qf, cf = F.extract_decision(G, p, cands, sim.nc, h_map, 0.0, 0.0,\n"
     "                                    ttl_const=TTL)\n",
     "        qf, cf = F.extract_decision(\n"
     "            G, p, cands, sim.nc, h_map, 0.0, 0.0, ttl_const=TTL,\n"
     "            own_queue_occ=(sim.queues[cur].occupancy\n"
     "                           if sim.schema_version >= 6 else None))\n", 1),
    ("def run_episode(cfg, scen_cfg, rate, seed, actor, model=None, device='cpu',\n"
     "                mask=None):\n",
     "def run_episode(cfg, scen_cfg, rate, seed, actor, model=None, device='cpu',\n"
     "                mask=None, schema_version=None):\n", 1),
    ("    sim = ModelActorSimulator(config, model, device, mask=mask)\n",
     "    sim = ModelActorSimulator(config, model, device, mask=mask,\n"
     "                              schema_version=schema_version)\n", 1),
    ("    ds = PhaseB(args.data, mask=mask_names)\n"
     "    assert_mask_applied(mask_names, args.device)\n",
     "    # v26: this legacy G4 script reads the v5 (Dataset V2) data deliberately\n"
     "    ds = PhaseB(args.data, mask=mask_names, accept_legacy=True)\n"
     "    assert_mask_applied(mask_names, args.device, schema_version=ds.schema_version)\n", 1),
    ("                        'schema': F.FEATURE_SCHEMA_VERSION,\n",
     "                        'schema': ds.schema_version,\n", 1),
    ("                m = run_episode(cfg, cfg, rate, s, None, model, args.device,\n"
     "                                mask=mask_names)\n",
     "                m = run_episode(cfg, cfg, rate, s, None, model, args.device,\n"
     "                                mask=mask_names,\n"
     "                                schema_version=ds.schema_version)\n", 2),
]

GENV2_EDITS = [
    ("ORACLE_TEACHER = 'spbp'                # G3: wins all 12 cells\n",
     "# LEGACY (Dataset V2 / schema v5) ONLY. The old comment here said SP-BP wins all\n"
     "# 12 cells; the oracle panels displaced it in all 9 oracle cells, and Dataset V3\n"
     "# labels with da_gpsr (generate_dataset_v3.py). Changing this constant would NOT\n"
     "# change the label: spbp_pick_restricted below is called unconditionally.\n"
     "ORACLE_TEACHER = 'spbp'\n", 1),
    ("        self.nc = F.norm_constants(config)\n",
     "        self.nc = F.norm_constants(config, schema_version=5)   # v26: legacy v5 data\n", 1),
    ("        'query_features': F.QUERY_FEATURES,\n",
     "        'query_features': F.LEGACY_FEATURE_LISTS[5]['query_features'],\n", 1),
    ("        'feature_schema_version': F.FEATURE_SCHEMA_VERSION,\n",
     "        'feature_schema_version': 5,          # v26: this generator is legacy v5\n", 1),
    ("        'norm_constants_per_scenario': {k: F.norm_constants({**BASE, **v})\n",
     "        'norm_constants_per_scenario': {k: F.norm_constants({**BASE, **v},\n"
     "                                                            schema_version=5)\n", 1),
    ("    ap.add_argument('--measure_only', action='store_true')\n"
     "    args = ap.parse_args()\n",
     "    ap.add_argument('--measure_only', action='store_true')\n"
     "    ap.add_argument('--legacy_v5', action='store_true',\n"
     "                    help='required: this generator is superseded by '\n"
     "                         'generate_dataset_v3.py and only reproduces the v5 dataset')\n"
     "    args = ap.parse_args()\n"
     "    if not args.legacy_v5:\n"
     "        raise SystemExit('generate_dataset_v2.py is LEGACY (Dataset V2, schema v5, SP-BP '\n"
     "                         'labels, 40 s). Use src/generate_dataset_v3.py; pass '\n"
     "                         '--legacy_v5 only to reproduce the old dataset.')\n", 1),
]

AUDITV2_EDITS = [
    ("    _skew = F.assert_manifest_compatible(man, context='audit')\n",
     "    _skew = F.assert_manifest_compatible(man, context='audit', accept_legacy=True)\n"
     "    QF_NAMES = list(man.get('query_features', F.QUERY_FEATURES))   # v26: dataset's own names\n", 1),
    ("    bad_q = col_report('query (per-decision)', qf, F.QUERY_FEATURES)\n",
     "    bad_q = col_report('query (per-decision)', qf, QF_NAMES)\n", 1),
    ("    q_cols = {nm: j for j, nm in enumerate(F.QUERY_FEATURES)}\n",
     "    q_cols = {nm: j for j, nm in enumerate(QF_NAMES)}\n", 1),
    ("            print(\"      -> weakly regime-dependent at this episode length. Note\")\n"
     "            print(\"         that the packet is dequeued BEFORE the decision is\")\n"
     "            print(\"         recorded, so this measures OTHER packets waiting at the\")\n"
     "            print(\"         current node, not the packet being routed.\")\n",
     "            print(\"      -> weakly regime-dependent at this episode length. CORRECTED\")\n"
     "            print(\"         (v26): the near-zero own-queue values are caused by\")\n"
     "            print(\"         _build_graph stamping queue_occupancy at FRAME START,\")\n"
     "            print(\"         before that frame's traffic exists -- not by the dequeue\")\n"
     "            print(\"         order. Dataset V3 (schema v6) reads the own queue live.\")\n", 1),
]

PREFLIGHTV2_EDITS = [
    ("    skew = F.assert_manifest_compatible(man, context='G3.5')\n",
     "    skew = F.assert_manifest_compatible(man, context='G3.5', accept_legacy=True)\n"
     "    QF_NAMES = list(man.get('query_features', F.QUERY_FEATURES))   # v26: dataset's own names\n", 1),
    ("    dead_query = [F.QUERY_FEATURES[i] for i, s in enumerate(qf_std) if s < 1e-9]\n",
     "    dead_query = [QF_NAMES[i] for i, s in enumerate(qf_std) if s < 1e-9]\n", 1),
    ("            ('query', qf, F.QUERY_FEATURES),\n",
     "            ('query', qf, QF_NAMES),\n", 2),
]

BAND_CC_EDITS = [
    ("    ap.add_argument('--initial_energy', type=float, default=None)\n"
     "    ap.add_argument('--max_workers', type=int, default=None)\n",
     "    ap.add_argument('--initial_energy', type=float, default=None)\n"
     "    # v26: altitude flags. Without them every run used BASE's 50-150 m, so the\n"
     "    # sink_50 band in results/band_sink50_1000s.json is at the wrong altitude.\n"
     "    # Defaults match find_usable_band.py and config_v2.OPERATING_POINT.\n"
     "    ap.add_argument('--z_min', type=float, default=100)\n"
     "    ap.add_argument('--z_max', type=float, default=300)\n"
     "    ap.add_argument('--max_workers', type=int, default=None)\n", 1),
    ("    cfg = suite[args.scenario]\n",
     "    cfg = {**suite[args.scenario], 'z_min': args.z_min, 'z_max': args.z_max}\n", 1),
    ("    print(f\"  initial_energy: {args.initial_energy if args.initial_energy is not None else 'DEFAULT (100.0)'}\")\n",
     "    print(f\"  initial_energy: {args.initial_energy if args.initial_energy is not None else 'DEFAULT (100.0)'}\")\n"
     "    print(f\"  altitude: {args.z_min:.0f}-{args.z_max:.0f} m (sink pinned at z_min)\")\n", 1),
    ("                   'run_params': {'duration': args.duration,\n"
     "                                  'initial_energy': args.initial_energy,\n",
     "                   'run_params': {'duration': args.duration,\n"
     "                                  'initial_energy': args.initial_energy,\n"
     "                                  'z_min': args.z_min, 'z_max': args.z_max,\n", 1),
]


def file_specs():
    return [
        ('config_v2.py', CONFIG_GUARD, 'append', [("def provenance():", None, 1)]),
        ('features_v2.py', 'BUFFERED_REF_V6 =', 'replace', features_edits()),
        ('train_supervised_v2.py', 'lossf_w = nn.CrossEntropyLoss', 'replace', TRAIN_EDITS),
        ('rollout_eval_v2.py', 'self.schema_version = int(F.FEATURE_SCHEMA_VERSION', 'replace',
         ROLLOUT_EDITS),
        ('generate_dataset_v2.py', "'--legacy_v5'", 'replace', GENV2_EDITS),
        ('audit_dataset_v2.py', 'QF_NAMES = list(', 'replace', AUDITV2_EDITS),
        ('preflight_dataset_v2_check.py', 'QF_NAMES = list(', 'replace', PREFLIGHTV2_EDITS),
        ('find_congestion_band_convergecast.py', "ap.add_argument('--z_min'", 'replace',
         BAND_CC_EDITS),
    ]


def stage(src, buffered_measured, buffered_use):
    staged, ok, notes = {}, True, []
    for fname, guard, kind, edits in file_specs():
        path = os.path.join(src, fname)
        if not os.path.isfile(path):
            notes.append(f'  {fname:<38} MISSING  <-- ABORT'); ok = False
            continue
        text = io.open(path, encoding='utf-8').read()
        if guard in text:
            notes.append(f'  {fname:<38} already applied (guard found) -- skipped')
            continue
        new = text
        if kind == 'append':
            anchor = edits[0][0]
            n = new.count(anchor)
            if n != 1:
                notes.append(f'  {fname:<38} anchor matched {n}x, expected 1  <-- ABORT'); ok = False
                continue
            new = new.rstrip('\n') + '\n' + CONFIG_BLOCK
        else:
            for i, (old, rep, want) in enumerate(edits, 1):
                if fname == 'features_v2.py':
                    rep = (rep.replace('{BUFFERED_REF_V6}', repr(float(BUFFERED_REF_V6)))
                           .replace('{BUFFERED_MEASURED}', buffered_measured)
                           .replace('{BUFFERED_USE}', buffered_use))
                n = new.count(old)
                if n != want:
                    notes.append(f'  {fname:<38} edit {i}: anchor matched {n}x, '
                                 f'expected {want}  <-- ABORT')
                    ok = False
                    break
                new = new.replace(old, rep)
        if new != text:
            staged[path] = new
            notes.append(f'  {fname:<38} {len(edits)} edit(s) staged')
    return staged, ok, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    staged, ok, notes = stage(a.src, BUFFERED_MEASURED, BUFFERED_USE)
    print('\n'.join(notes))
    if not ok:
        print('\n  NO FILE WRITTEN.')
        return 1
    if not staged:
        print('\n  ALREADY APPLIED. Nothing to do.')
        return 0
    if a.dry_run:
        print(f'\n  DRY RUN OK -- {len(staged)} file(s) would be written.')
        return 0
    for path, text in staged.items():
        with io.open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(text)
    print(f'\n  APPLIED to {len(staged)} file(s). Next: python verify_dataset_v3_v26.py --src {a.src}')
    return 0


# measured normaliser (filled from the 1000 s measurement, see features_v2 comment)
BUFFERED_REF_V6 = 300.0
BUFFERED_MEASURED = ('209 buffered packets (medium_slow@80; p99.9 at most 171 over the\n'
                     '# top band rate of all five scenarios, one 1000 s seed each)')
BUFFERED_USE = '42%'

if __name__ == '__main__':
    sys.exit(main())
