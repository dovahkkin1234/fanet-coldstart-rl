"""apply_grid_followup_v29.py -- the dataset grid follows the bands re-measured on the fixed
simulator, plus the follow-ups queued since v28.

    python apply_grid_followup_v29.py --root . --dry-run
    python apply_grid_followup_v29.py --root .
    python verify_grid_followup_v29.py --root .

WHY. The five bands were re-measured with the v28 mobility (results/band_*_v28.json,
2026-10-06). dense_slow, very_dense and medium_slow are unchanged; sparse_fast gained rate
60 and sink_50 gained rate 40. By the 2026-10-02 rule (band = the usable rates, anchor =
the lowest swept rate, below the band) DATASET_GRID follows: 18 cells, 900 episodes.

THIS PATCH
  src/config_v2.py                   DATASET_GRID: sparse_fast band 60/80/100, sink_50 band
                                     30/40 (changes the code signature: shards written with
                                     the old grid are regenerated, never resumed)
  src/rollout_gate_v3.py             the held-out medium_slow cells are rolled out and
                                     reported by default, as spec §8 says (--skip_heldout to
                                     opt out); a held-out cell is never gated, also when it is
                                     named in --cells. The verdict rule is unchanged.
  src/verify_dataset_grid_v3.py      docstring: the da_gpsr diagnostic numbers measured on v28
  verify_dataset_v3_v26.py           its config check expects the 18-cell grid
  verify_mobility_fix_v28.py         check 6 uses the newest band files as test data and
                                     compares the BAND MOVED report with the rule instead of
                                     expecting none (it would otherwise fail on any later grid)
  .gitignore                         smoke-test folders; raw episode recordings of the tests
  docs/DATASET_V3_SPEC.md, docs/CLAUDE.md, docs/FANET_Full_Project_Report.md,
  results/README.md                  the v28 results (bands, grid, teacher choice, Point-1),
                                     the 18-cell grid, the pre-v28 files marked as a record

Assertion-guarded str.replace: every anchor must match exactly once in every file, or
NOTHING is written. Idempotent: a file whose guard string is present is skipped. Line
endings: read in universal-newline mode, written as LF.
"""
import argparse
import io
import os
import sys

# ── src/config_v2.py ─────────────────────────────────────────────────────────
CONFIG = [
    ("# sink_50's band file was measured at 50-150 m (BASE); verify_dataset_grid_v3\n"
     "# marks it UNVERIFIED until it is re-measured at 100-300 m, and the generator\n"
     "# refuses unverified cells unless explicitly allowed.\n",
     "# v29 (2026-10-06): the bands were re-measured with the v28 mobility\n"
     "# (results/band_*_v28.json). dense_slow, very_dense and medium_slow are unchanged;\n"
     "# sparse_fast gained 60 (its elasticity fell from 0.86, against the ceiling, to 0.55 --\n"
     "# congestion-limited like the others once its drones stopped freezing) and sink_50\n"
     "# gained 40. The grid follows: 18 cells. verify_dataset_grid_v3 must PASS every cell\n"
     "# (and records BAND MOVED when this table and a band file disagree).\n", 1),
    ("    'sparse_fast': {'suite': 'A', 'anchor': [40.0], 'band': [80.0, 100.0]},\n"
     "    'sink_50':     {'suite': 'C', 'anchor': [20.0], 'band': [30.0]},\n",
     "    'sparse_fast': {'suite': 'A', 'anchor': [40.0], 'band': [60.0, 80.0, 100.0]},\n"
     "    'sink_50':     {'suite': 'C', 'anchor': [20.0], 'band': [30.0, 40.0]},\n", 1),
]

# ── src/rollout_gate_v3.py ──────────────────────────────────────────────────
GATE = [
    ("             is above -1.0 pp. Held-out medium_slow cells are REPORTED, not gated.\n",
     "             is above -1.0 pp. Held-out medium_slow cells are REPORTED, not gated\n"
     "             (v29: rolled out by default; --skip_heldout to leave them out).\n", 1),
    ("    ap.add_argument('--include_heldout', action='store_true')\n",
     "    ap.add_argument('--include_heldout', action='store_true',\n"
     "                    help='no-op since v29: held-out cells are rolled out and reported by default')\n"
     "    ap.add_argument('--skip_heldout', action='store_true',\n"
     "                    help='do not roll out the held-out cells (they are reported, never gated)')\n", 1),
    ("    cells = [(s, r) for s, r in C.dataset_cells() if s != C.GENERALISATION_SCENARIO]\n"
     "    if args.cells:\n"
     "        cells = [(s, r) for s, r in C.dataset_cells() if f'{s}@{r:g}' in args.cells]\n"
     "    held = ([(s, r) for s, r in C.dataset_cells() if s == C.GENERALISATION_SCENARIO]\n"
     "            if args.include_heldout else [])\n",
     "    # v29: the held-out cells are reported by default (spec §8) and never gated -- also\n"
     "    # when named in --cells (before v29 that made them gated)\n"
     "    cells = [(s, r) for s, r in C.dataset_cells() if s != C.GENERALISATION_SCENARIO]\n"
     "    held = [(s, r) for s, r in C.dataset_cells() if s == C.GENERALISATION_SCENARIO]\n"
     "    if args.cells:\n"
     "        cells = [(s, r) for s, r in cells if f'{s}@{r:g}' in args.cells]\n"
     "        held = [(s, r) for s, r in held if f'{s}@{r:g}' in args.cells]\n"
     "    if args.skip_heldout:\n"
     "        held = []\n", 1),
]

# ── src/verify_dataset_grid_v3.py ───────────────────────────────────────────
GRIDV = [
    ("measured 2026-10-02: dense_slow@60 0.019 and sink_50@30 0.015 overflow under\n"
     "da_gpsr vs 0.099 / 0.144 under the reference. Reported, never used to move cells.\n",
     "measured 2026-10-02: dense_slow@60 0.019 and sink_50@30 0.015 overflow under\n"
     "da_gpsr vs 0.099 / 0.144 under the reference; 2026-10-06 (v28 mobility): 0.019 and\n"
     "0.006 vs 0.096 / 0.036. Reported, never used to move cells.\n", 1),
]

# ── verify_dataset_v3_v26.py (repo root) ────────────────────────────────────
V26 = [
    ("        assert len(cells) == 16, len(cells)\n",
     "        assert len(cells) == 18, len(cells)      # v29: the grid follows the v28 bands (was 16)\n", 1),
    ("        return f'16 cells, BASE/RATES unchanged, explicit operating point{note}'\n",
     "        return f'{len(cells)} cells, BASE/RATES unchanged, explicit operating point{note}'\n", 1),
]

# ── verify_mobility_fix_v28.py (repo root), check 6 ─────────────────────────
V28 = [
    ("        covered = []\n        for sc, names in V.BAND_FILES.items():\n",
     "        covered, src_curves = [], {}\n        for sc, names in V.BAND_FILES.items():\n", 1),
    ("            have = [n for n in names[1:] if os.path.isfile(os.path.join(root, 'results', n))]\n"
     "            if not have:\n"
     "                continue\n"
     "            b = json.load(open(os.path.join(root, 'results', have[0])))\n"
     "            b.setdefault('run_params', {}).pop('mobility', None)\n"
     "            json.dump(b, open(os.path.join(res_old, have[0]), 'w'))\n",
     "            # v29: test data = the newest band file present (once re-measured, the v28 one,\n"
     "            # which matches the current grid); without its mobility marker, under the\n"
     "            # pre-v28 name, it plays a pre-v28 file\n"
     "            have = [n for n in names if os.path.isfile(os.path.join(root, 'results', n))]\n"
     "            if not have:\n"
     "                continue\n"
     "            b = json.load(open(os.path.join(root, 'results', have[0])))\n"
     "            b.setdefault('run_params', {}).pop('mobility', None)\n"
     "            json.dump(b, open(os.path.join(res_old, names[1]), 'w'))\n"
     "            src_curves[sc] = b['curve']\n", 1),
    ("        assert g_new.get('band_moved') == {}, f\"current bands reported as moved: {g_new.get('band_moved')}\"\n",
     "        # v29: the BAND MOVED report must equal the rule applied to the test data ({} when\n"
     "        # DATASET_GRID follows those bands) -- restated here, not taken from the verifier\n"
     "        chosen = {k.split('@')[0] for k, v in g_new['cells'].items() if 'band_file' in v}\n"
     "        expect = {}\n"
     "        for sc in chosen:\n"
     "            swept = sorted(float(c['rate']) for c in src_curves[sc])\n"
     "            use = sorted(float(c['rate']) for c in src_curves[sc] if c.get('usable'))\n"
     "            rule = {'anchor': [swept[0]] if use and swept[0] < use[0] else [], 'band': use}\n"
     "            gr = C.DATASET_GRID[sc]\n"
     "            if rule != {'anchor': [float(x) for x in gr['anchor']], 'band': [float(x) for x in gr['band']]}:\n"
     "                expect[sc] = rule\n"
     "        got = {sc: v['by_rule'] for sc, v in g_new.get('band_moved', {}).items()}\n"
     "        assert got == expect, f'BAND MOVED report {got} != the rule {expect}'\n", 1),
    ("                f'rest for other reasons, e.g. altitude), no band reported as moved{moved}; the generator '\n",
     "                f'rest for other reasons, e.g. altitude), BAND MOVED report = the rule '\n"
     "                f'({len(expect)} moved){moved}; the generator '\n", 1),
]

# ── .gitignore ──────────────────────────────────────────────────────────────
GITIGNORE = [
    ("results/m4_rollout_det2/\n",
     "results/m4_rollout_det2/\n\n"
     "# v29: smoke-test outputs (numbers meaningless) and the raw episode recordings the\n"
     "# 300 s tests regenerate exactly from their seeds (~100 MB per run)\n"
     "results/*_smoke/\n"
     "results/*/raw/\n", 1),
]

# ── docs/DATASET_V3_SPEC.md ─────────────────────────────────────────────────
SPEC = [
    ('— then **invalidated by the waypoint-trapping bug (§0 #15), fixed in v28: re-measure the bands '
     'and regenerate before any use (§10).**',
     '— then **invalidated by the waypoint-trapping bug (§0 #15), fixed in v28.** v29 (2026-10-06): '
     'the bands were re-measured on the fixed simulator and `DATASET_GRID` follows them (18 cells, §2); '
     'teacher choice and Point-1 were re-run. **Regenerate before any use (§10).**', 1),
    ('holding medium_slow out costs nothing (−0.04 pp) |',
     'holding medium_slow out costs nothing (−0.04 pp). **Re-run on the fixed simulator (v28, '
     '2026-10-05): same verdict** — dijkstra student −4.75 pp; mixing +0.12 / +0.03 pp; a da_gpsr '
     'student vs its teacher +0.47 / −0.05 / −0.01 pp; holdout −0.00 pp |', 1),
    ('| sparse_fast (flagged) | A | 40 | 80, **100** | PASS |\n'
     '| sink_50 | C | 20 | **30** | **UNVERIFIED** — band measured at 50–150 m |\n',
     '| sparse_fast | A | 40 | 60, 80, **100** | PASS (60 added in v29) |\n'
     '| sink_50 | C | 20 | 30, **40** | PASS (40 added in v29) |\n', 1),
    ("* **sink_50:** re-measure first (§10 step 2). If rate 30 is not usable at 100–300 m, update\n"
     "  `DATASET_GRID['sink_50']` by the same rule (band = usable rates, anchor = lowest swept) and\n"
     "  re-run the verifier before generating.\n",
     "* **sink_50:** measured at 100–300 m twice — before v28 only rate 30 was usable; on the fixed\n"
     "  simulator 30 and 40 are (v29).\n", 1),
    ('If a band moves, `DATASET_GRID` follows it by the rule above.\n',
     'If a band moves, `DATASET_GRID` follows it by the rule above.\n'
     '* **v29 (2026-10-06):** the five bands re-measured on the fixed simulator '
     '(`results/band_*_v28.json`, phase 1, 3 map seeds): dense_slow, very_dense and medium_slow '
     'unchanged; **sparse_fast gained 60** — its elasticity fell from 0.80–0.86 (against the 0.85 '
     'ceiling, near-zero link errors: unreachable destinations in frozen, partitioned networks) to '
     '0.45–0.55 with queue overflow 0.17–0.26, i.e. it is now congestion-limited like the other '
     'four; **sink_50 gained 40**. `DATASET_GRID` follows: 18 cells, 900 episodes. Buckets are '
     'scenario-relative, so sink_50@30 is now *medium*. medium_slow\'s top swept rate (80) is still '
     'usable, so its band\'s upper edge remains unmeasured. Re-run the verifier on the 18-cell grid '
     'before generating.\n', 1),
    ('The v28 run will differ (the networks keep moving).\n',
     'The v28 run will differ (the networks keep moving).\n\n'
     '**v29 grid (18 cells, 900 episodes):** sparse_fast@60 (~2–3 min per episode) and sink_50@40 '
     '(~15 min) add about 1.2 h of generation and ~12% more contexts. The 16-cell export at '
     '`--ctx_frac 0.6` was 20.2 GB, so the 18-cell one will likely exceed `--max_gb 22`: the exporter '
     'refuses before writing anything and prints the size — then use `--ctx_frac` ≈ 0.6 × 21.5 / '
     'that size. Point-1 measured delivery flat from 13 k to 1.7 M rows (its pre-registered reading, '
     'DATA-LIMITED, rests on hard-row accuracy — §12 item 8), so the fraction is set by RAM.\n', 1),
    ('**v28 re-run** (after `apply_mobility_fix_v28.py` and `verify_mobility_fix_v28.py`):\n\n'
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
     'Recommended before step 4: `test_teacher_choice.py --out results\\teacher_choice_v28` (~3 h). Its '
     'verdict (single da_gpsr) chose the label teacher, and it ran at 300 s, where 12–19% of the slow '
     'drones were trapped by the end of an episode. Optional after it: `test_bc_data_scaling.py --tc '
     'results\\teacher_choice_v28 --out results\\bc_data_scaling_v28` (~2 h, Point-1 on the fixed '
     'simulator).\n\n',
     '**v28/v29 re-run** (after `apply_mobility_fix_v28.py` / `verify_mobility_fix_v28.py`, then '
     '`apply_grid_followup_v29.py` / `verify_grid_followup_v29.py`):\n\n'
     '```\n'
     '1  DONE 2026-10-06: bands, phase 1 only, into results\\band_*_v28.json (--skip-self-test: the\n'
     '     self-tests check the code, not the band; verify_mobility_fix_v28 check 5 runs them)\n'
     '2  DONE: verify_dataset_grid_v3.py -> 16/16 PASS, BAND MOVED sparse_fast (+60), sink_50 (+40)\n'
     '     -> v29: DATASET_GRID follows (18 cells). Re-run on the 18-cell grid (18/18 PASS):\n'
     '     python src\\verify_dataset_grid_v3.py     (seconds; add --diagnose_seeds 3 --max_workers 12\n'
     '     for da_gpsr\'s congestion per cell, ~1 h -- reported, never used to move cells)\n'
     '3  python src\\generate_dataset_v3.py --smoke ; G3.5 v3 + audit on data\\v3_smoke\n'
     '4  python src\\generate_dataset_v3.py --out data\\v3m --max_workers 12   (a NEW folder; ~11 h)\n'
     '5  python src\\preflight_dataset_v3_check.py --data data\\v3m ; python src\\audit_dataset_v3.py --data data\\v3m\n'
     '6  python src\\export_phaseb_v3.py --data data\\v3m --out data\\phaseB_v3m --splits train val --ctx_frac 0.6\n'
     '     (refused above --max_gb 22, with the size printed: then --ctx_frac = 0.6 x 21.5 / size)\n'
     '7  python src\\rollout_gate_v3.py --data data\\phaseB_v3m --out results\\g4_v3m --max_workers 12\n'
     '     (v29: the held-out medium_slow cells are rolled out and reported by default)\n'
     '```\n'
     'Teacher choice on the fixed simulator: DONE 2026-10-05 (`results\\teacher_choice_v28`) — SINGLE '
     'da_gpsr, unchanged (§0 #1). Point-1: DONE 2026-10-06 (`results\\bc_data_scaling_v28`) — '
     'DATA-LIMITED again, on the same weak trigger (§12 item 8).\n\n', 1),
    ('| the 1000 s bands; "at 1000 s only rate 30 is usable in sink_50"; the v26 dataset and its per-cell PDRs |',
     '| sparse_fast is connectivity-limited; its band cells are not peers of the other scenarios\' '
     '(`results/README.md` caveat) | `results/README.md` | the trapping bug: on the fixed simulator it is '
     'congestion-limited (elasticity 0.45–0.55, queue overflow 0.17–0.26; v29) |\n'
     '| the 1000 s bands; "at 1000 s only rate 30 is usable in sink_50"; the v26 dataset and its per-cell PDRs |', 1),
    ('1. sink_50 band at 100–300 m (§10 step 2) → then the Suite C training weight (V3-10).\n',
     '1. sink_50 band at 100–300 m: done (v28 mobility: 30 and 40 usable, v29). The Suite C training '
     'weight (V3-10) remains open.\n', 1),
    ('7. **v28 re-run** (§10): bands → grid (`DATASET_GRID` may move) → regeneration → gates → export → G4.\n',
     '7. **v28 re-run** (§10): bands and grid done (v29: 18 cells), teacher choice and Point-1 re-run '
     '→ regeneration → gates → export → G4 (held-out cells reported by default).\n', 1),
    ('delivery is flat from 13.7 k to 1.8 M rows. Report both (V3-12).',
     'delivery is flat from 13.7 k to 1.8 M rows. Report both (V3-12). **Re-run on the fixed simulator '
     '(2026-10-06, `results/bc_data_scaling_v28`): DATA-LIMITED again, on the same single trigger** — '
     'very_dense hard rows +1.01 pp (30% → 100%), smaller than the two model seeds\' spread at that '
     'point (1.92 pp); delivery −0.04 pp (se 0.03); the 4× step lowered hard-row accuracy in all three '
     'cells (−0.28 / −0.08 / −0.53 pp); delivery vs da_gpsr flat from 13 k to 1.74 M rows (dense '
     '−0.02…+0.06, very_dense −0.08…−0.01 pp). New on the moving networks: small-data students are '
     'worse on hard rows (very_dense at 3%: 92.1% vs 95.0% before the fix), converging by 100%. The '
     'medium_slow student-vs-teacher offset (+0.5–0.6 pp in every run) is not significant (p 0.12–0.18, '
     'n = 5).', 1),
]

# ── docs/CLAUDE.md ──────────────────────────────────────────────────────────
CLAUDE_MD = [
    ('networks. Next blocking pieces, in order: re-measure the five bands with `--phase1_only`,\n'
     '`verify_dataset_grid_v3.py` (DATASET_GRID follows a band that moved), re-confirm the\n'
     'teacher choice (`test_teacher_choice.py` into a new folder -- its 300 s verdict chose the\n'
     'label teacher), regenerate into a new folder, G3.5 v3 + audit v3, export, then the\n'
     'retrain gate. Never resume',
     'networks. v29 (2026-10-06): on the fixed simulator the five bands were re-measured\n'
     '(sparse_fast gained 60, sink_50 gained 40; DATASET_GRID follows -- 18 cells), the teacher\n'
     'choice re-confirmed (SINGLE da_gpsr) and Point-1 re-run (DATA-LIMITED on the same weak\n'
     'trigger). Next blocking pieces, in order: `verify_dataset_grid_v3.py` on the 18-cell grid,\n'
     'smoke, regenerate into a new folder (data\\v3m), G3.5 v3 + audit v3, export, then the\n'
     'retrain gate (held-out cells reported by default). Never resume', 1),
]

# ── docs/FANET_Full_Project_Report.md ───────────────────────────────────────
REPORT = [
    ('The bands are re-measured after the fix; treat these as pre-v28.',
     'The bands are re-measured after the fix; treat these as pre-v28.\n>\n'
     '> **v29 (2026-10-06): re-measured on the fixed simulator** (`results/band_*_v28.json`) — '
     'dense_slow 60/80/100, very_dense 60/80 and medium_slow 40/60/80 unchanged; sparse_fast '
     '60/80/100 (was 80/100: elasticity 0.45–0.55 instead of 0.80–0.86, i.e. congestion-limited, '
     'not partitioned); sink_50 30/40 (was 30). `DATASET_GRID` follows (18 cells).', 1),
    ('> analysis below is the reasoning before the test. See DATASET_V3_SPEC §0-§1.',
     '> analysis below is the reasoning before the test. See DATASET_V3_SPEC §0-§1.\n>\n'
     '> **Re-run on the fixed simulator (v28 mobility, 2026-10-05, `results/teacher_choice_v28`): '
     'SINGLE da_gpsr again** — dijkstra student −4.75 pp vs a da_gpsr student in medium_slow; mixing '
     'teachers +0.12 / +0.03 pp in the dense cells (n.s.); dijkstra\'s medium_slow edge (+2.78 pp) is '
     'its dropping of unreachable packets (without it: −0.23 pp vs da_gpsr, n.s.).', 1),
]

# ── results/README.md ───────────────────────────────────────────────────────
RES_README = [
    ('on the re-measured band before citing it.\n',
     'on the re-measured band before citing it.\n\n'
     '> **v29 (2026-10-06).** Bands, grid verification, teacher choice and Point-1 were re-measured '
     'on the fixed simulator; the CURRENT table lists those files, and the pre-v28 ones moved to '
     '**PRE-V28** below (kept as the record). The sparse_fast caveat was the bug: on the fixed '
     'simulator sparse_fast is congestion-limited like the others.\n', 1),
    ('| `band_dense_slow_1000s.json` | Usable band + queue ablation, 1000 s, battery 8000. Usable rates 60/80/100. |\n'
     '| `band_very_dense_1000s.json` | Same, very_dense. Usable rates 60/80. |\n'
     '| `band_medium_slow_1000s.json` | Same, medium_slow. Usable rates 40/60/80. |\n'
     '| `band_sparse_fast_1000s.json` | Same, sparse_fast. Usable rates 80/100 — **see caveat below**. |\n',
     '| `band_dense_slow_1000s_v28.json` | Usable band (phase 1), 1000 s, battery 8000, 100–300 m, v28 mobility. Usable rates 60/80/100. |\n'
     '| `band_very_dense_1000s_v28.json` | Same, very_dense. Usable rates 60/80 (100 and 120 lose ~8 nodes to the battery). |\n'
     '| `band_medium_slow_1000s_v28.json` | Same, medium_slow. Usable rates 40/60/80; 80 is the top of the sweep, so the upper edge is unmeasured. |\n'
     '| `band_sparse_fast_1000s_v28.json` | Same, sparse_fast. Usable rates 60/80/100 — congestion-limited (see the resolved caveat below). |\n'
     '| `band_sink50_1000s_z100_300_v28.json` | Convergecast band, same operating point. Usable rates 30/40. |\n', 1),
    ('| `band_sink50_1000s.json` | Convergecast band, 1000 s, battery 8000. Only rate 30 usable. **Altitude 50–150 m '
     '(BASE), not the 100–300 m operating point** — Dataset V3 needs `band_sink50_1000s_z100_300.json` '
     '(DATASET_V3_SPEC §2). |\n', '', 1),
    ('| `teacher_choice/summary.json` | Teacher-choice test (2026-10-01): pre-registered verdict SINGLE da_gpsr. '
     '300 s, 100–300 m, battery 8000, 10 paired seeds. |\n'
     '| `grid_verification.json` | Dataset V3 grid gate (`verify_dataset_grid_v3.py`); the generator refuses cells '
     'that are not PASS here. |\n'
     '| `bc_data_scaling/scaling.json` | How much imitation data is worth training on (`test_bc_data_scaling.py`). |\n'
     '| `dataset_v3/<data>_preflight_v3.json`, `_audit_v3.json`, `_manifest.json` | Dataset V3 gate verdicts (G3.5 '
     'v3, independent audit) and manifest, copied out of the git-ignored `data/` folder. |\n'
     '| `g4_v3/gate.json` | Restated G4 check 4 (`rollout_gate_v3.py`): student vs restricted da_gpsr, paired '
     'non-inferiority, 1 pp margin. |\n',
     '| `teacher_choice_v28/summary.json` | Teacher-choice test on the fixed simulator (2026-10-05): pre-registered '
     'verdict SINGLE da_gpsr, unchanged. 300 s, 100–300 m, battery 8000, 10 paired seeds. |\n'
     '| `grid_verification.json` | Dataset V3 grid gate (`verify_dataset_grid_v3.py`), v28 mobility; the generator '
     'refuses cells that are not PASS here and treats a file from another mobility version as verifying nothing. '
     'Re-run after any `DATASET_GRID` change (v29: 18 cells). |\n'
     '| `bc_data_scaling_v28/scaling.json` | Point-1 on the fixed simulator (2026-10-06): DATA-LIMITED on the same '
     'single trigger as before (DATASET_V3_SPEC §12 item 8). |\n', 1),
    ('### Caveat on `band_sparse_fast_1000s.json`\n\n'
     'Its two "usable" cells have elasticity **0.80 and 0.85**, both pressed against\n'
     'the `ELASTIC_HI = 0.85` ceiling, while every other scenario\'s usable cells sit\n'
     'at **0.05–0.47**. Its `link_error` is ~0.007 (near zero) against a PDR of only\n'
     '~0.30, so its dominant loss is neither link nor queue — most likely\n'
     'route-unavailability, consistent with the original connectivity-limited\n'
     'characterisation of this scenario. Treat its cells as qualitatively different\n'
     'from the other four scenarios\', not as peers.\n',
     '### sparse_fast caveat — resolved by v28\n\n'
     'Before v28, sparse_fast\'s "usable" cells had elasticity 0.80–0.86, pressed\n'
     'against the `ELASTIC_HI = 0.85` ceiling, with near-zero link errors (~0.007) and\n'
     'little queue overflow: its loss was neither link nor queue but unreachable\n'
     'destinations. That was the trapping bug — by 300 s every sparse_fast drone was\n'
     'frozen and the network stayed partitioned. On the fixed simulator its usable\n'
     'cells have elasticity 0.45–0.55 and queue overflow 0.17–0.26: it is\n'
     'congestion-limited like the other four scenarios, and its cells are peers of theirs.\n\n'
     '## PRE-V28 — the trapping simulator (kept as the record, not current)\n\n'
     '| file | what it is | current replacement |\n'
     '|---|---|---|\n'
     '| `band_dense_slow_1000s.json`, `band_very_dense_1000s.json`, `band_medium_slow_1000s.json`, '
     '`band_sparse_fast_1000s.json` | 1000 s bands with the phase-2 queue ablation | `band_*_1000s_v28.json` (phase 1) |\n'
     '| `band_sink50_1000s_z100_300.json` | sink_50 band at 100–300 m, with the phase-2 ablation (candqueue +3.51 pp, '
     'additive +4.08 pp) | `band_sink50_1000s_z100_300_v28.json` |\n'
     '| `band_sink50_1000s.json` | sink_50 band at 50–150 m (BASE), not the operating point | — |\n'
     '| `teacher_choice/` | teacher-choice test, 2026-10-01: SINGLE da_gpsr | `teacher_choice_v28/` |\n'
     '| `bc_data_scaling/scaling.json` | Point-1, 2026-10-04: DATA-LIMITED | `bc_data_scaling_v28/` |\n'
     '| `dataset_v3/v3_*` | Dataset V3 v26 run: G3.5 v3 and audit PASS | the regeneration into `data\\v3m` |\n'
     '| `g4_v3/gate.json` | restated G4 on the v26 dataset: PASS, held-out cells inside the margin | `g4_v3m/` |\n', 1),
]


def file_specs():
    return [
        ('src/config_v2.py', "'sink_50':     {'suite': 'C', 'anchor': [20.0], 'band': [30.0, 40.0]}", CONFIG),
        ('src/rollout_gate_v3.py', "'--skip_heldout'", GATE),
        ('src/verify_dataset_grid_v3.py', '2026-10-06 (v28 mobility)', GRIDV),
        ('verify_dataset_v3_v26.py', 'assert len(cells) == 18', V26),
        ('verify_mobility_fix_v28.py', 'covered, src_curves = [], {}', V28),
        ('.gitignore', 'results/*/raw/', GITIGNORE),
        ('docs/DATASET_V3_SPEC.md', '* **v29 (2026-10-06):**', SPEC),
        ('docs/CLAUDE.md', 'v29 (2026-10-06): on the fixed simulator', CLAUDE_MD),
        ('docs/FANET_Full_Project_Report.md', '**v29 (2026-10-06): re-measured on the fixed simulator**', REPORT),
        ('results/README.md', '## PRE-V28 — the trapping simulator', RES_README),
    ]


def stage(root):
    staged, ok, notes = {}, True, []
    for rel, guard, edits in file_specs():
        path = os.path.join(root, *rel.split('/'))
        if not os.path.isfile(path):
            notes.append(f'  {rel:<36} MISSING  <-- ABORT')
            ok = False
            continue
        text = io.open(path, encoding='utf-8').read()
        if guard in text:
            notes.append(f'  {rel:<36} already applied (guard found) -- skipped')
            continue
        new = text
        for i, (old, rep, want) in enumerate(edits, 1):
            n = new.count(old)
            if n != want:
                notes.append(f'  {rel:<36} edit {i}: anchor matched {n}x, expected {want}  <-- ABORT')
                ok = False
                break
            new = new.replace(old, rep)
        else:
            if guard not in new:
                notes.append(f'  {rel:<36} guard missing after edits  <-- ABORT (patch bug)')
                ok = False
                continue
            staged[path] = new
            notes.append(f'  {rel:<36} {len(edits)} edit(s) staged')
    return staged, ok, notes


def main():
    ap = argparse.ArgumentParser(description='v29 grid follow-up (see module docstring)')
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
    print(f'\n  APPLIED to {len(staged)} file(s). Next: python verify_grid_followup_v29.py --root {a.root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
