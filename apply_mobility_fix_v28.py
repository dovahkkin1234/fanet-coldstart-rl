"""apply_mobility_fix_v28.py -- the waypoint-trapping fix, and the guards that keep
pre-fix and post-fix results apart.

    python apply_mobility_fix_v28.py --root . --dry-run
    python apply_mobility_fix_v28.py --root .
    python verify_mobility_fix_v28.py --root .

THE BUG (since the first commit). mobility.DroneRWP.step moves a drone by
speed * dt every 0.5 s frame (2.5-25 m) but counts an arrival only within 1 m of
the waypoint. A step that passed the waypoint without landing within 1 m of it
overshot, turned back and overshot again, every frame, forever: the drone never
picked a new waypoint. Trapped at 1000 s: 37-51% of the 5-15 m/s drones, every
sparse_fast and sink_50 drone -- frozen in place, jumping a full step every frame.

THIS PATCH
  src/mobility.py                 the fix (only that step changes) + MOBILITY_VERSION
  src/find_usable_band.py         --phase1_only; band files record the mobility version
  src/find_congestion_band_convergecast.py   same
  src/verify_dataset_grid_v3.py   refuses band files from another mobility version and
                                  prints the five re-measure commands
  src/generate_dataset_v3.py      CODE SIGNATURE of every module that shapes an episode,
                                  recorded per shard and in the manifest; resume and the
                                  manifest ignore shards written by other code
  src/preflight_dataset_v3_check.py  check 0: the dataset must come from the current code,
                                  and every shard from the same code
  src/export_phaseb_v3.py         refuses a dataset from other code (--allow_stale_code)
  src/rollout_gate_v3.py          refuses an export from other code (--allow_stale_code)
  src/test_teacher_choice.py      records the mobility version; never resumes a folder
                                  written by another version
  src/test_bc_data_scaling.py     refuses to replay or pair with teacher-choice results
                                  from another mobility version (hard/curve/analyze still run)
  docs/DATASET_V3_SPEC.md, docs/CLAUDE.md, docs/FANET_Full_Project_Report.md,
  results/README.md               the finding, the re-run order, what is now pre-v28

Assertion-guarded str.replace: every anchor must match exactly once (or the stated
count) in every file, or NOTHING is written. Idempotent: a file whose guard string is
present is skipped. Line endings: read in universal-newline mode, written as LF.
"""
import argparse
import io
import os
import sys

MOB = 'v28-arrive-on-pass'

# ─────────────────────────────────────────────────────────────────────────────
MOBILITY = [
    ('"""\nmobility.py\nRandom Waypoint 3D mobility model for FANET simulation.\n\n'
     'Each drone moves independently. No coordination between drones.\n"""\n\n'
     'import numpy as np\n',
     '"""\nmobility.py\nRandom Waypoint 3D mobility model for FANET simulation.\n\n'
     'Each drone moves independently. No coordination between drones.\n\n'
     'v28 (2026-10-04): waypoint-trapping fix -- see DroneRWP.step. MOBILITY_VERSION\n'
     'is recorded by the band scripts, the dataset generator and the tests; the grid\n'
     'and dataset gates refuse results produced with a different version.\n"""\n\n'
     'import numpy as np\n\n'
     f"MOBILITY_VERSION = '{MOB}'\n", 1),
    ('        self.vz = dz * scale * vertical_fraction\n\n'
     '        self.x += self.vx * dt\n',
     '        self.vz = dz * scale * vertical_fraction\n\n'
     '        # v28 FIX -- WAYPOINT TRAPPING. A step is speed * dt (2.5-25 m at the\n'
     '        # 0.5 s frame) but arrival needs dist < 1 m. A step that passed the\n'
     '        # waypoint without landing within 1 m of it overshot, turned back and\n'
     '        # overshot again, every frame, forever: the drone never picked a new\n'
     '        # waypoint. Measured at 1000 s: 37-51% of the 5-15 m/s drones and every\n'
     '        # sparse_fast / sink_50 drone frozen in place, jumping a full step every\n'
     '        # frame (fake velocities, flickering links). Such a step now ARRIVES.\n'
     '        # Every other step is the original code, so a trajectory is unchanged\n'
     "        # up to the drone's first would-be overshoot.\n"
     '        sx, sy, sz = self.vx * dt, self.vy * dt, self.vz * dt\n'
     '        if sx * sx + sy * sy + sz * sz >= dist * dist:\n'
     '            rx, ry, rz = dx - sx, dy - sy, dz - sz\n'
     '            if rx * rx + ry * ry + rz * rz >= 1.0:\n'
     '                self.x, self.y, self.z = self.dest_x, self.dest_y, self.dest_z\n'
     '                self.vx, self.vy, self.vz = dx / dt, dy / dt, dz / dt\n'
     '                self.pause_remaining = self.rng.uniform(0, self.pause_max)\n'
     '                self._pick_new_waypoint()\n'
     '                return\n\n'
     '        self.x += self.vx * dt\n', 1),
]

BAND_COMMON = [
    ('from simulator_v2 import FANETSimulatorV2\n',
     'from simulator_v2 import FANETSimulatorV2\n'
     'from mobility import MOBILITY_VERSION                  # v28: recorded in run_params\n', 1),
    ("    ap.add_argument('--skip-self-test', action='store_true')\n",
     "    ap.add_argument('--skip-self-test', action='store_true')\n"
     "    ap.add_argument('--phase1_only', action='store_true',\n"
     "                    help='v28: stop after phase 1 (which rates are usable) -- all the '\n"
     "                         'dataset grid needs')\n", 1),
    ("                                  'measure_seeds': args.measure_seeds,\n",
     "                                  'measure_seeds': args.measure_seeds,\n"
     "                                  'phase1_only': args.phase1_only,\n"
     "                                  'mobility': MOBILITY_VERSION,\n", 1),
]
USABLE_BAND = BAND_COMMON + [
    ('    results = {}\n    if usable:\n        seeds = list(range(1, args.measure_seeds + 1))\n',
     '    results = {}\n    if usable and args.phase1_only:\n'
     "        print('\\n  --phase1_only: phase 2 (matched queue ablation) skipped')\n"
     '    elif usable:\n        seeds = list(range(1, args.measure_seeds + 1))\n', 1),
]
CONVERGECAST = BAND_COMMON + [
    ("    results = {}\n    if not usable:\n",
     "    results = {}\n    if usable and args.phase1_only:\n"
     "        print('  ' + ', '.join(f\"rate {c['rate']:.1f}\" for c in usable))\n"
     "        print('  --phase1_only: phase 2 (matched queue ablation) skipped')\n"
     "    elif not usable:\n", 1),
]

GRID = [
    ('               (duration 1000 s, battery 8000, altitude 100-300 m) and marks the\n',
     '               (duration 1000 s, battery 8000, altitude 100-300 m, and -- v28 --\n'
     '               the current MOBILITY_VERSION) and marks the\n', 1),
    ('import config_v2 as C                                        # noqa: E402\n',
     'import config_v2 as C                                        # noqa: E402\n'
     'from mobility import MOBILITY_VERSION                        # noqa: E402\n', 1),
    ("BAND_FILES = {\n"
     "    'dense_slow': ['band_dense_slow_1000s.json'],\n"
     "    'very_dense': ['band_very_dense_1000s.json'],\n"
     "    'medium_slow': ['band_medium_slow_1000s.json'],\n"
     "    'sparse_fast': ['band_sparse_fast_1000s.json'],\n"
     "    # the re-measurement at 100-300 m goes FIRST; the original (50-150 m) is the fallback\n"
     "    'sink_50': ['band_sink50_1000s_z100_300.json', 'band_sink50_1000s.json'],\n}\n",
     "BAND_FILES = {\n"
     "    # v28: the re-measurement with the fixed mobility goes FIRST (suffix _v28, so the\n"
     "    # pre-v28 files are kept for comparison); those stay listed only so the verifier\n"
     "    # can say why it refuses them\n"
     "    'dense_slow': ['band_dense_slow_1000s_v28.json', 'band_dense_slow_1000s.json'],\n"
     "    'very_dense': ['band_very_dense_1000s_v28.json', 'band_very_dense_1000s.json'],\n"
     "    'medium_slow': ['band_medium_slow_1000s_v28.json', 'band_medium_slow_1000s.json'],\n"
     "    'sparse_fast': ['band_sparse_fast_1000s_v28.json', 'band_sparse_fast_1000s.json'],\n"
     "    # the re-measurement at 100-300 m goes FIRST; the original (50-150 m) is the fallback\n"
     "    'sink_50': ['band_sink50_1000s_z100_300_v28.json', 'band_sink50_1000s_z100_300.json',\n"
     "                'band_sink50_1000s.json'],\n}\n\n"
     "# v28: the commands that produce each band file at the operating point with the\n"
     "# current mobility. Phase 1 decides which rates are usable -- all the grid needs.\n"
     "# --skip-self-test: the self-tests check the CODE (determinism, paired traffic, no\n"
     "# collapse) in 8 serial full-length episodes -- hours at 1000 s;\n"
     "# verify_mobility_fix_v28.py check 5 runs them on both band scripts.\n"
     "REMEASURE = {\n"
     "    'dense_slow': r'python src\\find_usable_band.py --scenario dense_slow --duration 1000 '\n"
     "                  r'--initial_energy 8000 --rates 40 60 80 100 120 160 --phase1_only --skip-self-test '\n"
     "                  r'--max_workers 12 --out results\\band_dense_slow_1000s_v28.json',\n"
     "    'very_dense': r'python src\\find_usable_band.py --scenario very_dense --duration 1000 '\n"
     "                  r'--initial_energy 8000 --rates 40 60 80 100 120 --phase1_only --skip-self-test '\n"
     "                  r'--max_workers 12 --out results\\band_very_dense_1000s_v28.json',\n"
     "    'medium_slow': r'python src\\find_usable_band.py --scenario medium_slow --duration 1000 '\n"
     "                   r'--initial_energy 8000 --rates 30 40 60 80 --phase1_only --skip-self-test '\n"
     "                   r'--max_workers 12 --out results\\band_medium_slow_1000s_v28.json',\n"
     "    'sparse_fast': r'python src\\find_usable_band.py --scenario sparse_fast --duration 1000 '\n"
     "                   r'--initial_energy 8000 --rates 40 60 80 100 120 --phase1_only --skip-self-test '\n"
     "                   r'--max_workers 12 --out results\\band_sparse_fast_1000s_v28.json',\n"
     "    'sink_50': r'python src\\find_congestion_band_convergecast.py --scenario sink_50 '\n"
     "               r'--duration 1000 --initial_energy 8000 --rates 20 30 40 50 60 --phase1_only --skip-self-test '\n"
     "               r'--max_workers 8 --out results\\band_sink50_1000s_z100_300_v28.json',\n"
     "}\n", 1),
    ("    out = {'operating_point': C.OPERATING_POINT, 'grid': C.DATASET_GRID, 'cells': {},\n",
     "    out = {'operating_point': C.OPERATING_POINT, 'grid': C.DATASET_GRID, 'cells': {},\n"
     "           'mobility': MOBILITY_VERSION, 'band_moved': {},\n", 1),
    ("            chosen = (fn, b)\n            break\n        for r in C.dataset_rates(sc):\n",
     "            chosen = (fn, b)\n            break\n"
     "        if chosen is not None:\n"
     "            # v28: the grid follows the band (decision 2026-10-02: band = the usable rates,\n"
     "            # anchor = the lowest swept rate, below the band). Reported here, never applied.\n"
     "            swept = sorted(float(c['rate']) for c in chosen[1]['curve'])\n"
     "            use = sorted(float(c['rate']) for c in chosen[1]['curve'] if c.get('usable'))\n"
     "            rule = {'anchor': [swept[0]] if use and swept[0] < use[0] else [], 'band': use}\n"
     "            if rule != {'anchor': [float(x) for x in g['anchor']], 'band': [float(x) for x in g['band']]}:\n"
     "                out['band_moved'][sc] = {'grid': {'anchor': g['anchor'], 'band': g['band']},\n"
     "                                         'by_rule': rule, 'band_file': chosen[0]}\n"
     "        for r in C.dataset_rates(sc):\n", 1),
    ("            'z': z}\n",
     "            'z': z, 'mobility': rp.get('mobility')}\n", 1),
    ("    if op['initial_energy'] != o['initial_energy']:\n"
     "        why.append(f\"battery {op['initial_energy']} != {o['initial_energy']}\")\n",
     "    if op['initial_energy'] != o['initial_energy']:\n"
     "        why.append(f\"battery {op['initial_energy']} != {o['initial_energy']}\")\n"
     "    if op.get('mobility') != MOBILITY_VERSION:\n"
     "        why.append(f\"mobility {op.get('mobility') or 'pre-v28 (waypoint trapping)'} \"\n"
     "                   f\"!= {MOBILITY_VERSION}\")\n", 1),
    ("        print('  sink_50: re-measure its band at 100-300 m (the band script now has --z_min/--z_max):')\n"
     "        print('    python src\\\\find_congestion_band_convergecast.py --scenario sink_50 --duration 1000 '\n"
     "              '--initial_energy 8000 --rates 20 30 40 50 60 --measure-seeds 15 --max_workers 8 '\n"
     "              '--out results\\\\band_sink50_1000s_z100_300.json')\n",
     "    todo = [sc for sc in REMEASURE if any(v['status'] == 'UNVERIFIED' for k, v in out['cells'].items()\n"
     "                                          if k.split('@')[0] == sc)]\n"
     "    if todo:\n"
     "        print('  Measure these bands at the operating point with the current mobility:')\n"
     "        for sc in todo:\n"
     "            print('    ' + REMEASURE[sc])\n"
     "    for sc, v in out['band_moved'].items():\n"
     "        r = v['by_rule']\n"
     "        note = ('  -- no usable rate: widen --rates' if not r['band'] else\n"
     "                '  -- no swept rate below the band: add a lower rate' if not r['anchor'] else '')\n"
     "        print(f\"  BAND MOVED  {sc}: DATASET_GRID {v['grid']}  ->  by the rule {r}  ({v['band_file']}){note}\")\n"
     "    if out['band_moved']:\n"
     "        print('  DATASET_GRID in config_v2 must follow the band before generating (decision '\n"
     "              '2026-10-02: band = the usable rates, anchor = the lowest swept rate, below the band).')\n", 1),
]

GENERATOR = [
    ('from generate_dataset_v2 import canonical_candidates          # noqa: E402\n',
     'from generate_dataset_v2 import canonical_candidates          # noqa: E402\n'
     'from mobility import MOBILITY_VERSION                          # noqa: E402\n', 1),
    ('assert MAX_QUEUE < OWNQ_BASE <= 127\n',
     'assert MAX_QUEUE < OWNQ_BASE <= 127\n\n'
     '# v28: CODE SIGNATURE. A shard is valid only for the code that wrote it: every\n'
     "# module that shapes an episode (simulator, mobility, link model, teachers,\n"
     '# features, config, this generator). Recorded in every shard and the manifest;\n'
     '# resume, the manifest, G3.5 v3, the export and the gate compare it with the\n'
     '# current code. Line endings are normalised (git on Windows rewrites LF/CRLF).\n'
     'SRC_DIR = os.path.dirname(os.path.abspath(__file__))\n'
     "SIGNATURE_MODULES = ('config_v2', 'features_v2', 'generate_dataset_v2', 'generate_dataset_v3',\n"
     "                     'link_model_v2', 'mobility', 'models', 'routing_teachers',\n"
     "                     'routing_teachers_v2', 'routing_teachers_v3_local', 'simulator_v2',\n"
     "                     'teacher_panel', 'teacher_pickers_v3')\n\n\n"
     'def code_signature(src_dir=SRC_DIR):\n'
     '    h = hashlib.sha256()\n'
     '    for name in SIGNATURE_MODULES:\n'
     "        h.update(name.encode() + b'\\0')\n"
     "        with open(os.path.join(src_dir, name + '.py'), 'rb') as f:\n"
     "            h.update(f.read().replace(b'\\r\\n', b'\\n'))\n"
     '    return h.hexdigest()[:16]\n\n\n'
     'CODE_SIGNATURE = code_signature()\n', 1),
    ("        'complete': True,\n        'record_schema_version': RECORD_SCHEMA_VERSION,\n",
     "        'complete': True,\n        'record_schema_version': RECORD_SCHEMA_VERSION,\n"
     "        'code_signature': CODE_SIGNATURE, 'mobility': MOBILITY_VERSION,\n", 1),
    ("        and 'n_ownq_pairs' in m.get('counts', {})\n",
     "        and 'n_ownq_pairs' in m.get('counts', {}) and m.get('code_signature') == CODE_SIGNATURE\n", 1),
    ("                        and 'n_ownq_pairs' in m.get('counts', {})):\n",
     "                        and 'n_ownq_pairs' in m.get('counts', {})\n"
     "                        and m.get('code_signature') == CODE_SIGNATURE):\n", 1),
    ("        'generator_sha256_16': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16],\n",
     "        'code_signature': CODE_SIGNATURE, 'mobility': MOBILITY_VERSION,\n"
     "        'generator_sha256_16': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16],\n", 1),
    ("    metas = []\n    root = os.path.join(out, 'shards')\n",
     "    metas, skipped = [], []\n    root = os.path.join(out, 'shards')\n", 1),
    ("                else:\n"
     "                    print(f'  manifest: skipping incomplete or stale shard {f}')\n"
     "    metas.sort(key=lambda m: (m['scenario'], m['rate'], m['seed']))\n",
     "                else:\n"
     "                    skipped.append(f)\n"
     "                    print(f'  manifest: skipping incomplete or stale shard {f}')\n"
     "    if skipped and not metas:\n"
     "        # v28: rebuilding the manifest of a pre-v28 dataset would silently empty it\n"
     "        raise SystemExit(f'  manifest NOT written: all {len(skipped)} shards in {out} are incomplete '\n"
     "                         f'or written by other code (current signature {CODE_SIGNATURE}). '\n"
     "                         f'Regenerate into a new --out; the existing manifest.json is untouched.')\n"
     "    metas.sort(key=lambda m: (m['scenario'], m['rate'], m['seed']))\n", 1),
    ("def load_grid_verification(path):\n"
     "    if not os.path.isfile(path):\n"
     "        return None\n"
     "    return json.load(open(path))\n",
     "def load_grid_verification(path):\n"
     "    if not os.path.isfile(path):\n"
     "        return None\n"
     "    return json.load(open(path))\n\n\n"
     "def unverified_cells(gv, cells):\n"
     "    \"\"\"Cells that are not PASS in the grid verification. v28: a verification written\n"
     "    for another mobility version verifies nothing -- its bands were measured on\n"
     "    another simulator.\"\"\"\n"
     "    gv = gv or {}\n"
     "    out = []\n"
     "    for s, r in cells:\n"
     "        st = gv.get('cells', {}).get(f'{s}@{r:g}', {}).get('status')\n"
     "        if st == 'PASS' and gv.get('mobility') != MOBILITY_VERSION:\n"
     "            st = f\"PASS-but-mobility-{gv.get('mobility') or 'pre-v28'}\"\n"
     "        if st != 'PASS':\n"
     "            out.append(f'{s}@{r:g}={st or \"missing\"}')\n"
     "    return out\n", 1),
    ("    gv = load_grid_verification(args.grid_verification)\n"
     "    unverified = []\n"
     "    for s, r in cells:\n"
     "        st = (gv or {}).get('cells', {}).get(f'{s}@{r:g}', {}).get('status')\n"
     "        if st != 'PASS':\n"
     "            unverified.append(f'{s}@{r:g}={st or \"missing\"}')\n",
     "    gv = load_grid_verification(args.grid_verification)\n"
     "    unverified = unverified_cells(gv, cells)\n", 1),
    (r"""            '\n  Run src\\verify_dataset_grid_v3.py first (sink_50 needs its band '
            're-measured at 100-300 m), or pass --allow_unverified_cells to '
""",
     r"""            '\n  Run src\\verify_dataset_grid_v3.py first (v28: every band is re-measured '
            'with the current mobility -- the verifier prints the commands), or pass '
            '--allow_unverified_cells to '
""", 1),
    ("    print('  self-test: canonical cache exact, v6 refuses a missing live own-queue, '\n"
     "          'packet sample nested  OK')\n",
     "    print('  self-test: canonical cache exact, v6 refuses a missing live own-queue, '\n"
     "          'packet sample nested  OK')\n"
     "    # (multiprocessing aliases __main__ as __mp_main__ in every process that imports it)\n"
     "    stray = sorted(n for n, mod in list(sys.modules.items())\n"
     "                   if n not in ('__main__', '__mp_main__') and getattr(mod, '__file__', None)\n"
     "                   and os.path.dirname(os.path.abspath(mod.__file__)) == SRC_DIR\n"
     "                   and n not in SIGNATURE_MODULES)\n"
     "    if stray:\n"
     "        raise SystemExit(f'  the code signature misses modules the generator loaded: {stray} '\n"
     "                         f'-- add them to SIGNATURE_MODULES')\n"
     "    print(f'  code signature {CODE_SIGNATURE} (mobility {MOBILITY_VERSION}): shards written '\n"
     "          f'by other code are regenerated, never resumed')\n", 1),
]

PREFLIGHT = [
    ("    p += F.assert_manifest_compatible(man, context='G3.5v3')\n    return p\n",
     "    if man.get('code_signature') != G.CODE_SIGNATURE:\n"
     "        p.append(f\"dataset written by other code: signature \"\n"
     "                 f\"{man.get('code_signature') or 'none (pre-v28)'} != current \"\n"
     "                 f\"{G.CODE_SIGNATURE} -- regenerate it (v28: mobility fix)\")\n"
     "    p += F.assert_manifest_compatible(man, context='G3.5v3')\n    return p\n\n\n"
     "def check_signatures(shard_sigs, man):\n"
     "    \"\"\"v28: every shard must come from the code the manifest names.\"\"\"\n"
     "    bad = sorted({str(s) for s in shard_sigs if s != man.get('code_signature')})\n"
     "    return [f'shards written by other code than the manifest: {bad}'] if bad else []\n", 1),
    ('    rng = np.random.default_rng(1)\n    nc_ok = True\n',
     '    rng = np.random.default_rng(1)\n    nc_ok = True\n    shard_sigs = []\n', 1),
    ("            nc_ok = False\n        eps_n += meta['counts']['n_recorded_decisions']\n",
     "            nc_ok = False\n        shard_sigs.append(meta.get('code_signature'))\n"
     "        eps_n += meta['counts']['n_recorded_decisions']\n", 1),
    ("        details['0 schema'].append('norm constants differ between episodes of a scenario')\n",
     "        details['0 schema'].append('norm constants differ between episodes of a scenario')\n"
     "    sig_probs = check_signatures(shard_sigs, man)\n"
     "    if sig_probs:\n"
     "        results['0 schema'] = False\n"
     "        details['0 schema'] += sig_probs\n", 1),
]

EXPORT = [
    ("    ap.add_argument('--force', action='store_true')\n",
     "    ap.add_argument('--force', action='store_true')\n"
     "    ap.add_argument('--allow_stale_code', action='store_true',\n"
     "                    help='export a dataset written by other code (recorded in the manifest)')\n", 1),
    ('                         f"histograms cannot be exported)")\n',
     '                         f"histograms cannot be exported)")\n'
     "    if man.get('code_signature') != G.CODE_SIGNATURE:\n"
     "        msg = (f\"dataset written by other code (signature {man.get('code_signature') or 'none, pre-v28'} \"\n"
     "               f\"!= current {G.CODE_SIGNATURE}): its episodes do not come from this simulator\")\n"
     "        if not args.allow_stale_code:\n"
     "            raise SystemExit(msg + ' -- regenerate, or pass --allow_stale_code')\n"
     "        print('  WARNING: ' + msg)\n", 1),
    ("        'record_schema_version': man['record_schema_version'],\n",
     "        'record_schema_version': man['record_schema_version'],\n"
     "        'code_signature': man.get('code_signature'), 'mobility': man.get('mobility'),\n"
     "        'stale_code_allowed': bool(args.allow_stale_code),\n", 1),
]

GATE = [
    ("    ap.add_argument('--stage', choices=('all', 'train', 'rollout', 'analyze'), default='all')\n",
     "    ap.add_argument('--stage', choices=('all', 'train', 'rollout', 'analyze'), default='all')\n"
     "    ap.add_argument('--allow_stale_code', action='store_true',\n"
     "                    help='gate an export whose dataset was written by other code')\n", 1),
    ("        raise SystemExit(f'eval seeds {bad} are dataset seeds -- that would measure memorisation')\n",
     "        raise SystemExit(f'eval seeds {bad} are dataset seeds -- that would measure memorisation')\n"
     "    if args.stage in ('all', 'train', 'rollout'):\n"
     "        # v28: the reference runs in THIS simulator; the student must have learned from it\n"
     "        import generate_dataset_v3 as G\n"
     "        pman = os.path.join(args.data, 'manifest.json')\n"
     "        sig = json.load(open(pman)).get('code_signature') if os.path.isfile(pman) else None\n"
     "        if sig != G.CODE_SIGNATURE and not args.allow_stale_code:\n"
     "            raise SystemExit(f'  export {args.data} comes from other code (signature {sig or \"none, pre-v28\"} '\n"
     "                             f'!= current {G.CODE_SIGNATURE}) -- re-export a regenerated dataset, '\n"
     "                             f'or pass --allow_stale_code')\n", 1),
]

SCALING = [
    ('sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n',
     'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n'
     'from mobility import MOBILITY_VERSION                     # noqa: E402  (v28)\n', 1),
    ('    s = {k: c.get(k, v) for k, v in TC_DEFAULTS.items()}\n',
     '    s = {k: c.get(k, v) for k, v in TC_DEFAULTS.items()}\n'
     "    s['mobility'] = c.get('mobility')                      # None = produced before v28\n", 1),
    ("    res = json.load(open(p)) if os.path.isfile(p) else {'hard': {}, 'curve': {}, 'pdr': {}}\n",
     "    res = json.load(open(p)) if os.path.isfile(p) else {'hard': {}, 'curve': {}, 'pdr': {}}\n"
     "    # v28: the extension replays the teacher-choice episodes and the rollouts are paired\n"
     "    # with its rollout.json -- both only valid in the simulator that produced them. A\n"
     "    # scaling.json belongs to the simulator of ONE teacher-choice run.\n"
     "    sim = a.stage in ('extend', 'rollout') or (a.stage == 'all' and (a.extend or a.rollout))\n"
     "    tc_mob = a.tc_cfg['mobility']\n"
     "    if sim and tc_mob != MOBILITY_VERSION:\n"
     "        raise SystemExit(f\"  {a.tc} was produced with mobility \"\n"
     "                         f\"{tc_mob or 'pre-v28 (waypoint trapping)'}; this simulator \"\n"
     "                         f'is {MOBILITY_VERSION}. Its episodes cannot be replayed or paired here: '\n"
     "                         f're-run test_teacher_choice.py into a new --out, or run only '\n"
     "                         f'--stage hard / curve / analyze.')\n"
     "    if any(res.get(k) for k in ('hard', 'curve', 'pdr')) and res.get('mobility') != tc_mob:\n"
     "        raise SystemExit(f\"  {p} holds results from mobility {res.get('mobility') or 'pre-v28'}, \"\n"
     "                         f\"{a.tc} is from {tc_mob or 'pre-v28'} -- use a new --out\")\n"
     "    res['mobility'] = tc_mob\n", 1),
]

TEACHER = [
    ('from generate_dataset_v2 import canonical_candidates\nimport inspect\n',
     'from generate_dataset_v2 import canonical_candidates\nimport inspect\n'
     'from mobility import MOBILITY_VERSION                     # v28\n', 1),
    ("    os.makedirs(args.out, exist_ok=True)\n    _save(args, 'config.json', vars(args))\n",
     "    os.makedirs(args.out, exist_ok=True)\n"
     "    # v28: never mix simulators in one output folder (the mobility fix changes every\n"
     "    # trajectory longer than ~30 s)\n"
     "    prev = _load(args, 'config.json', None)\n"
     "    if prev is not None and prev.get('mobility') != MOBILITY_VERSION:\n"
     "        raise SystemExit(f\"  {args.out} was produced with mobility \"\n"
     "                         f\"{prev.get('mobility') or 'pre-v28 (waypoint trapping)'}; this \"\n"
     "                         f'simulator is {MOBILITY_VERSION}. Use a new --out.')\n"
     "    args.mobility = MOBILITY_VERSION\n"
     "    _save(args, 'config.json', vars(args))\n", 1),
    ("    out = {'operating_point': {'duration': args.duration, 'z': [args.z_min, args.z_max],\n",
     "    out = {'operating_point': {'duration': args.duration, 'z': [args.z_min, args.z_max],\n"
     "                               'mobility': MOBILITY_VERSION,\n", 1),
]

# ── documents ────────────────────────────────────────────────────────────────
F15 = ("| 15 | **Drones get trapped at their waypoints** (found 2026-10-04, after the full v26 run; "
       "in `mobility.py` since the first commit) | a mobility step is speed × 0.5 s (2.5–25 m) but "
       "arrival needs < 1 m, so a step that passes the waypoint without landing within 1 m starts an "
       "endless overshoot oscillation. Trapped at 300 s / 1000 s: 12–19% / 37–51% of the 5–15 m/s "
       "drones, 100% / 100% of sparse_fast, 91% / 100% of sink_50; a trapped drone jumps a full step "
       "every frame (fake velocities, flickering links). In the v26 dataset 38–46% of sparse_fast "
       "seeds deliver < 5% of packets. Same episode with the fix: sparse_fast@40 seed 106 PDR 0.021 → "
       "0.352, seed 101 0.486 → 0.425; sink_50@30 seed 136 0.223 → 0.728 (link-error drops 55% → "
       "26%); dense_slow@60 seed 101 0.682 → 0.684. At the 40 s parity point the slow scenarios are "
       "essentially unchanged (≤ 0.14 pp; G2 anchors identical); sparse_fast is not (+10.8 pp on "
       "average) | v28: a step that would pass the waypoint arrives; `MOBILITY_VERSION` and a "
       "code signature are recorded by the band scripts, the shards and the tests, and every gate "
       "refuses a mismatch. Bands re-measured, dataset regenerated (§10) |")

SPEC = [
    ('**Status:** implemented (patch v26), smoke-tested; full generation not yet run.',
     '**Status:** v26 implemented and run in full (2026-10-04: 800 episodes, G3.5 v3 and audit PASS) '
     '— then **invalidated by the waypoint-trapping bug (§0 #15), fixed in v28: re-measure the bands '
     'and regenerate before any use (§10).**', 1),
    ('\n\n---\n\n## 1. Decisions (locked 2026-10-02 unless marked open)',
     '\n' + F15 + '\n\n---\n\n## 1. Decisions (locked 2026-10-02 unless marked open)', 1),
    ('  re-run the verifier before generating.\n\n---\n\n## 3. Files',
     '  re-run the verifier before generating.\n'
     '* **v28:** a band file counts only if it records `mobility = v28-arrive-on-pass`; the verifier '
     'refuses older files and prints the five re-measure commands (`--phase1_only`: which rates are '
     'usable is all the grid needs). They write `band_*_v28.json` beside the pre-v28 files, which '
     'are kept for comparison. `grid_verification.json` records the mobility too, and the generator '
     'treats a pre-v28 one as verifying nothing. If a band moves, `DATASET_GRID` follows it by the '
     'rule above.'
     '\n\n---\n\n## 3. Files', 1),
    ('| G3.5 v3 | `preflight_dataset_v3_check.py` | 0 schema ·',
     '| G3.5 v3 | `preflight_dataset_v3_check.py` | 0 schema (v28: and the code signature — the '
     'dataset must come from the current simulator / feature / generator code, every shard from the '
     'same code) ·', 1),
    ('Measured in the sandbox (60 s smoke episodes, 150 s probes, one 1000 s run per scenario) and\n'
     'extrapolated to 1000 s:',
     '**Actual v26 run** (2026-10-04, Z8, 12 workers, with the trapping bug): 10.0 h for 800 episodes; '
     '120.1 M contexts, 471 M own-queue pairs, 71.5 M steps, 714 M decisions, 11.5 GB. Export at '
     '`--ctx_frac 0.6` of train + val: 40.1 M rows, 997 k frames, 20.2 GB in 155 s. The v28 run will '
     'differ (the networks keep moving).\n\n'
     'Measured in the sandbox (60 s smoke episodes, 150 s probes, one 1000 s run per scenario) and\n'
     'extrapolated to 1000 s:', 1),
    ('## 10. Execution order\n\n',
     '## 10. Execution order\n\n'
     '**v28 re-run** (after `apply_mobility_fix_v28.py` and `verify_mobility_fix_v28.py`):\n\n'
     '```\n'
     '1  bands, phase 1 only (commands printed by verify_dataset_grid_v3.py; ~2-3 h):\n'
     '     find_usable_band.py x 4 (dense_slow, very_dense, medium_slow, sparse_fast) and\n'
     '     find_congestion_band_convergecast.py (sink_50), all --duration 1000 --initial_energy 8000,\n'
     '     into results\\band_*_v28.json (the pre-v28 files are kept); --skip-self-test, because\n'
     '     the self-tests check the code, not the band (verify_mobility_fix_v28 check 5 runs them)\n'
     '2  python src\\verify_dataset_grid_v3.py --diagnose_seeds 3 --max_workers 12\n'
     '     (BAND MOVED lines = a band changed -> DATASET_GRID follows it, then re-verify)\n'
     '3  python src\\generate_dataset_v3.py --smoke ; G3.5 v3 + audit on data\\v3_smoke\n'
     '4  python src\\generate_dataset_v3.py --out data\\v3m --max_workers 12   (a NEW folder)\n'
     '5  python src\\preflight_dataset_v3_check.py --data data\\v3m ; python src\\audit_dataset_v3.py --data data\\v3m\n'
     '6  python src\\export_phaseb_v3.py --data data\\v3m --out data\\phaseB_v3m --splits train val --ctx_frac 0.6\n'
     '7  python src\\rollout_gate_v3.py --data data\\phaseB_v3m --out results\\g4_v3m --max_workers 12\n'
     '```\n'
     'Recommended before step 4: `test_teacher_choice.py --out results\\teacher_choice_v28` (~3 h). '
     'Its verdict (single da_gpsr) chose the label teacher, and it ran at 300 s, where 12–19% of the '
     'slow drones were trapped by the end of an episode. Optional after it: '
     '`test_bc_data_scaling.py --tc results\\teacher_choice_v28 --out results\\bc_data_scaling_v28` '
     '(~2 h, Point-1 on the fixed simulator).\n\n'
     '**Original v26 order** (run 2026-10-03/04 on the trapping simulator; kept for the record):\n\n', 1),
    ('| G4 check 4 = ≥ 90% of SP-BP | rollout_eval_v2 | non-inferiority vs restricted da_gpsr (§8) |',
     '| G4 check 4 = ≥ 90% of SP-BP | rollout_eval_v2 | non-inferiority vs restricted da_gpsr (§8) |\n'
     '| drones follow Random Waypoint for the whole episode | `mobility.py` since the first commit | '
     'they froze at a waypoint after a missed arrival (§0 #15); fixed in v28 |\n'
     '| the 1000 s bands; "at 1000 s only rate 30 is usable in sink_50"; the v26 dataset and its '
     'per-cell PDRs | report §11.2, `results/band_*_1000s.json`, `data/v3` | measured on freezing '
     'networks; re-measured / regenerated after v28 |\n'
     '| link-lifetime saturation is purely structural (§0 #10) | this spec | partly confounded: a '
     'trapped drone reports a velocity that flips every frame; re-check on the v28 dataset |', 1),
    ("5. `BUFFERED_REF_V6` re-checked by G3.5 v3's saturation report on the full dataset.",
     "5. `BUFFERED_REF_V6`: checked on the v26 run (largest value 73% of the range, nothing clipped); "
     "re-confirm on the v28 run.", 1),
    ('6. The isolated-node queue quirk (§0 #13): fix or keep, before the M5 environment is frozen.',
     '6. The isolated-node queue quirk (§0 #13): fix or keep, before the M5 environment is frozen.\n'
     '7. **v28 re-run** (§10): bands → grid (`DATASET_GRID` may move) → regeneration → gates → export '
     '→ G4.\n'
     '8. Point-1 (`test_bc_data_scaling.py`, 2026-10-04): the pre-registered verdict is DATA-LIMITED, '
     'recorded as such. Its only trigger (very_dense hard rows +0.77 pp, 30% → 100%) is within '
     'model-seed noise (two seeds differ by up to 1.0 pp), delivery moved +0.04 pp, and the 4× step '
     'reversed it in every cell; delivery is flat from 13.7 k to 1.8 M rows. Report both (V3-12).', 1),
]

CLAUDE_MD = [
    ('As of v26 (2026-10-02): Dataset V3 is implemented (generator, gates, audit, export,\n'
     'restated G4) per `docs/DATASET_V3_SPEC.md`; feature schema is v6 (live own queue).\n'
     'Next blocking pieces, in order: re-measure the sink_50 band at 100-300 m, run\n'
     '`verify_dataset_grid_v3.py`, smoke + full generation, G3.5 v3 + audit v3, then the\n'
     'retrain gate.',
     'As of v28 (2026-10-04): the v26 dataset was generated and passed every gate, then the\n'
     'mobility bug was found (drones trapped at their waypoints; DATASET_V3_SPEC §0 #15) and\n'
     'fixed in v28. Results from episodes longer than ~40 s produced before v28 ran on freezing\n'
     'networks. Next blocking pieces, in order: re-measure the five bands with `--phase1_only`,\n'
     '`verify_dataset_grid_v3.py` (DATASET_GRID follows a band that moved), re-confirm the\n'
     'teacher choice (`test_teacher_choice.py` into a new folder -- its 300 s verdict chose the\n'
     'label teacher), regenerate into a new folder, G3.5 v3 + audit v3, export, then the\n'
     'retrain gate. Never resume or mix outputs across a `MOBILITY_VERSION` / code-signature\n'
     'change -- the scripts refuse it. Feature\n'
     'schema is v6 (live own queue).', 1),
]

REPORT = [
    ('Re-measure it before Dataset V3 uses it (DATASET_V3_SPEC §2).',
     'Re-measure it before Dataset V3 uses it (DATASET_V3_SPEC §2).\n\n'
     '> **v28 (2026-10-04): every number in this section was measured with the waypoint-trapping '
     'bug** — at 1000 s, 37–51% of the 5–15 m/s drones and every sparse_fast / sink_50 drone end the '
     'episode frozen at a waypoint (DATASET_V3_SPEC §0 #15). The bands are re-measured after the fix; '
     'treat these as pre-v28.', 1),
]

RESULTS_README = [
    ('side by\nside with no marking.\n',
     'side by\nside with no marking.\n\n'
     '> **v28 (2026-10-04) — mobility fix.** Every result from episodes longer than ~40 s produced '
     'before v28 ran on networks that freeze (drones trapped at their waypoints: 37–51% of the '
     '5–15 m/s drones and all sparse_fast / sink_50 drones by 1000 s; DATASET_V3_SPEC §0 #15). '
     'That covers the `band_*_1000s*.json` files, `grid_verification.json`, `teacher_choice/` '
     'and `bc_data_scaling/` (300 s), `dataset_v3/` and `g4_v3/`. Read them as pre-v28 until '
     're-measured; files written after v28 record `mobility: v28-arrive-on-pass` (the re-measured '
     'bands are `band_*_v28.json`; the pre-v28 band files are kept beside them). The 40 s results '
     'are essentially unaffected except in sparse_fast. The sparse_fast caveat below (low delivery, '
     'loss neither link nor queue) is what a frozen, partitioned network looks like — re-check it '
     'on the re-measured band before citing it.\n', 1),
]


def file_specs():
    return [
        ('src/mobility.py', 'MOBILITY_VERSION', MOBILITY),
        ('src/find_usable_band.py', "'mobility': MOBILITY_VERSION", USABLE_BAND),
        ('src/find_congestion_band_convergecast.py', "'mobility': MOBILITY_VERSION", CONVERGECAST),
        ('src/verify_dataset_grid_v3.py', 'REMEASURE = {', GRID),
        ('src/generate_dataset_v3.py', 'CODE_SIGNATURE = code_signature()', GENERATOR),
        ('src/preflight_dataset_v3_check.py', 'def check_signatures', PREFLIGHT),
        ('src/export_phaseb_v3.py', '--allow_stale_code', EXPORT),
        ('src/rollout_gate_v3.py', '--allow_stale_code', GATE),
        ('src/test_bc_data_scaling.py', 'from mobility import MOBILITY_VERSION', SCALING),
        ('src/test_teacher_choice.py', 'from mobility import MOBILITY_VERSION', TEACHER),
        ('docs/DATASET_V3_SPEC.md', '| 15 | **Drones get trapped', SPEC),
        ('docs/CLAUDE.md', 'As of v28 (2026-10-04)', CLAUDE_MD),
        ('docs/FANET_Full_Project_Report.md', '**v28 (2026-10-04): every number in this section', REPORT),
        ('results/README.md', '**v28 (2026-10-04) — mobility fix.**', RESULTS_README),
    ]


def stage(root):
    staged, ok, notes = {}, True, []
    for rel, guard, edits in file_specs():
        path = os.path.join(root, *rel.split('/'))
        if not os.path.isfile(path):
            notes.append(f'  {rel:<44} MISSING  <-- ABORT')
            ok = False
            continue
        text = io.open(path, encoding='utf-8').read()
        if guard in text:
            notes.append(f'  {rel:<44} already applied (guard found) -- skipped')
            continue
        new = text
        for i, (old, rep, want) in enumerate(edits, 1):
            n = new.count(old)
            if n != want:
                notes.append(f'  {rel:<44} edit {i}: anchor matched {n}x, expected {want}  <-- ABORT')
                ok = False
                break
            new = new.replace(old, rep)
        else:
            if guard not in new:
                notes.append(f'  {rel:<44} guard missing after edits  <-- ABORT (patch bug)')
                ok = False
                continue
            staged[path] = new
            notes.append(f'  {rel:<44} {len(edits)} edit(s) staged')
    return staged, ok, notes


def main():
    ap = argparse.ArgumentParser(description='v28 mobility fix (see module docstring)')
    ap.add_argument('--root', default='.')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    staged, ok, notes = stage(a.root)
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
    print(f'\n  APPLIED to {len(staged)} file(s). Next: python verify_mobility_fix_v28.py --root {a.root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
