"""apply_docs_v3_v27.py -- bring the documents in line with Dataset V3 (docs/DATASET_V3_SPEC.md).

    python apply_docs_v3_v27.py --root . --dry-run
    python apply_docs_v3_v27.py --root .

Same contract as the code patches: every anchor must match exactly once, all edits
are staged in memory, nothing is written unless everything matches, and a file whose
guard is already present is skipped (idempotent). Requires docs/DATASET_V3_SPEC.md
to exist (it is copied in with the v26 files).

  docs/DATASET_V2_MASTER_SPEC.md     SUPERSEDED banner
  docs/FILE2_PRE_M5_EXECUTION_PLAN.md banner: which sections are superseded (stride 10,
                                      rate grid, SP-BP oracle, D2 vs SP-BP)
  docs/FILE1_APPROACH2_RECORD.md     banner: SP-BP-oracle and link-lifetime statements
  docs/CLAUDE.md                     where-to-read pointers + "where things stand"
  docs/FANET_Full_Project_Report.md  panel altitude (50-150 m, not 100-300), sink_50
                                      band altitude, link-lifetime claim withdrawn,
                                      §7.6 superseded, §17.1 decided
  docs/Panel_Results_Report.md       altitude row corrected
  results/README.md                  sink_50 band altitude, panel altitude, new files
"""
import argparse
import io
import os
import sys

V2_SPEC = [("# Dataset V2 — Master Specification\n",
            "# Dataset V2 — Master Specification\n\n"
            "> **SUPERSEDED (2026-10-02) by `docs/DATASET_V3_SPEC.md`.** Kept as history. Its global\n"
            "> rate grid was voided by the v12 leak fix, its SP-BP oracle was displaced in all 9\n"
            "> oracle cells, and its action-consistency check (R-23) would fail on correct data.\n"
            "> DATASET_V3_SPEC §11 lists every statement here that no longer holds.\n", 1)]

FILE2 = [("# FILE 2 — PRE-M5 EXECUTION PLAN\n",
          "# FILE 2 — PRE-M5 EXECUTION PLAN\n\n"
          "> **PARTLY SUPERSEDED (2026-10-02).** For the dataset follow `docs/DATASET_V3_SPEC.md`;\n"
          "> for the overall order follow `docs/FANET_Full_Project_Report.md` §18. Superseded here:\n"
          "> the rate grids of §1.5 / §1.5.1 (void after v12; per-scenario band + anchor now),\n"
          "> the stride-10 patch of §2 (frames stride 1; packet-sampled steps; de-duplicated\n"
          "> contexts), the SP-BP oracle and D2's `+0.0645` SP-BP reference (oracle panels;\n"
          "> label teacher da_gpsr), and checkpoint-based mid-trajectory sampling as a dataset\n"
          "> feature. Still valid: the parity-gate-before-v8b ordering (§3) and the D1-D5 designs\n"
          "> (§9) subject to the restatements in the report's §19.2.\n", 1)]

FILE1 = [("# FILE 1 — APPROACH 2 RECORD: M3 → M4 COMPLETE\n",
          "# FILE 1 — APPROACH 2 RECORD: M3 → M4 COMPLETE\n\n"
          "> **Corrections (2026-10-02).** SP-BP is no longer the oracle (displaced in all 9\n"
          "> oracle cells; Dataset V3 labels with da_gpsr). The statement that the\n"
          "> `estimated_link_lifetime` saturation (60.4% at the 60 s cap) is a 40 s-episode\n"
          "> artifact is withdrawn: 45-73% of edges sit at the cap at every time of a 1000 s\n"
          "> episode because the estimator returns the cap for any non-separating pair.\n"
          "> Current record: `docs/FANET_Full_Project_Report.md`; dataset: `docs/DATASET_V3_SPEC.md`.\n", 1)]

CLAUDE = [
    ("FANET lifelong-RL routing research, IIITDM Kancheepuram. For project state, read\n"
     "`docs/FILE1_APPROACH2_RECORD.md` (what was built and measured) and\n"
     "`docs/FILE2_PRE_M5_EXECUTION_PLAN.md` (what's next, in order) — do not restate their\n"
     "content here; this file is rules, not knowledge.\n",
     "FANET lifelong-RL routing research, IIITDM Kancheepuram. For project state, read\n"
     "`docs/FANET_Full_Project_Report.md` (the current record and plan) and, for the dataset,\n"
     "`docs/DATASET_V3_SPEC.md`. `docs/FILE1_APPROACH2_RECORD.md` / `FILE2_PRE_M5_EXECUTION_PLAN.md`\n"
     "are history: their banners say what in them is superseded (FILE2's stride-10 patch,\n"
     "rate grids and SP-BP oracle must NOT be implemented). Do not restate their content\n"
     "here; this file is rules, not knowledge.\n", 1),
    ("Next blocking piece: `experiment_headroom.py` needs `--duration`/`--z_min`/`--z_max`/\n"
     "`--drain_time` CLI flags (v11.2, FILE2 §1.5.1) before the three-quantity rate probe can\n"
     "run. Do not apply `apply_sim_changes_v8.py` before the new RL environment's SP-BP-parity\n"
     "gate passes at the current 40 s operating point (FILE2 §3) — that gate is the only fixed\n"
     "reference for telling \"the environment is right\" apart from \"the operating point moved.\"",
     "As of v26 (2026-10-02): Dataset V3 is implemented (generator, gates, audit, export,\n"
     "restated G4) per `docs/DATASET_V3_SPEC.md`; feature schema is v6 (live own queue).\n"
     "Next blocking pieces, in order: re-measure the sink_50 band at 100-300 m, run\n"
     "`verify_dataset_grid_v3.py`, smoke + full generation, G3.5 v3 + audit v3, then the\n"
     "retrain gate. Labels come from da_gpsr through `teacher_pickers_v3` only — never add\n"
     "another label path. Training rows take `own_queue_live` from each context's own-queue\n"
     "histogram (`export_phaseb_v3.py`), never from `c_query` (first occurrence, biased high);\n"
     "an RL state built from a step uses `s_ownq`. Do not apply\n"
     "`apply_v8b_operating_point_STAGED.py` before the new\n"
     "RL environment's parity gate passes at the current 40 s point; the dataset does not\n"
     "need v8b (it uses `config_v2.OPERATING_POINT`).", 1),
]

REPORT = [
    ("All at 1000 s, battery 8000, altitude 100–300, **10 paired seeds per cell**",
     "All at 1000 s, battery 8000, **altitude 50–150 m** (config_v2.BASE: the panel scripts set "
     "duration and battery but never altitude — corrected 2026-10-02; the bands that chose these "
     "rates ran at 100–300 m, except sink_50), **10 paired seeds per cell**", 1),
    ("unblocks the `energy` feature (std 0.0205 at 40 s) and `estimated_link_lifetime` (60.4% at "
     "its cap because `LIFETIME_REF = 60 s` > 40 s) — **do not also change `LIFETIME_REF`** |",
     "unblocks the `energy` feature (std 0.0205 at 40 s). *Corrected 2026-10-02:* it does **not** "
     "unblock `estimated_link_lifetime` — 45–73% of edges sit at the 60 s cap at every time of a "
     "1000 s episode, because the estimator returns the cap for any non-separating pair. "
     "`LIFETIME_REF` stays 60 |", 1),
    ("### 11.2 Usable bands at 1000 s / battery 8000 [MEASURED, 15 paired seeds — `results/band_*_1000s.json`]",
     "### 11.2 Usable bands at 1000 s / battery 8000 [MEASURED, 15 paired seeds — `results/band_*_1000s.json`]\n\n"
     "> Altitude 100–300 m for the four Suite A scenarios; **sink_50's band was measured at "
     "50–150 m** (the convergecast band script had no altitude flags — added in v26). "
     "Re-measure it before Dataset V3 uses it (DATASET_V3_SPEC §2).", 1),
    ("### 7.6 The regenerated schema (specified, not implemented) — `M3.5_Dataset_Schema.md`\n",
     "### 7.6 The regenerated schema — superseded by `docs/DATASET_V3_SPEC.md` (implemented in v26)\n\n"
     "> The paragraph below is the pre-2026-10-02 plan. V3 changed: single da_gpsr label (no\n"
     "> `label_teacher`), schema v6, no `checkpoint_id`, `eps_fired` instead of the\n"
     "> `action != label` check, de-duplicated contexts + packet-sampled steps.\n\n", 1),
    ("### 17.1 One global teacher, or per-cell teachers? (you are researching this)\n",
     "### 17.1 One global teacher, or per-cell teachers? — **DECIDED 2026-10-01: single global da_gpsr**\n\n"
     "> Teacher-choice test, pre-registered verdict SINGLE da_gpsr: dijkstra labels are not\n"
     "> learnable locally (student −3.97 pp vs a da_gpsr student in medium_slow), mixing\n"
     "> teachers did not hurt the dense cells, a da_gpsr student matches its teacher. The\n"
     "> analysis below is the reasoning before the test. See DATASET_V3_SPEC §0-§1.\n", 1),
]

PANEL = [("| Altitude band | 100–300 m | matches published comparators |",
          "| Altitude band | **50–150 m** — config_v2.BASE (the panel scripts set duration and "
          "battery, never altitude; corrected 2026-10-02). The band search that chose the rates "
          "ran at 100–300 m, except sink_50 | the dataset operating point is 100–300 m |", 1)]

RESULTS = [
    ("| `band_sink50_1000s.json` | Convergecast band, 1000 s, battery 8000. Only rate 30 usable. |",
     "| `band_sink50_1000s.json` | Convergecast band, 1000 s, battery 8000. Only rate 30 usable. "
     "**Altitude 50–150 m (BASE), not the 100–300 m operating point** — Dataset V3 needs "
     "`band_sink50_1000s_z100_300.json` (DATASET_V3_SPEC §2). |", 1),
    ("| `energy_range.json` | Battery sweep that established `INITIAL_ENERGY = 8000`. |",
     "| `energy_range.json` | Battery sweep that established `INITIAL_ENERGY = 8000`. |\n"
     "| `teacher_choice/summary.json` | Teacher-choice test (2026-10-01): pre-registered verdict "
     "SINGLE da_gpsr. 300 s, 100–300 m, battery 8000, 10 paired seeds. |\n"
     "| `grid_verification.json` | Dataset V3 grid gate (`verify_dataset_grid_v3.py`); the generator "
     "refuses cells that are not PASS here. |\n"
     "| `bc_data_scaling/scaling.json` | How much imitation data is worth training on "
     "(`test_bc_data_scaling.py`). |\n"
     "| `dataset_v3/<data>_preflight_v3.json`, `_audit_v3.json`, `_manifest.json` | Dataset V3 "
     "gate verdicts (G3.5 v3, independent audit) and manifest, copied out of the git-ignored "
     "`data/` folder. |\n"
     "| `g4_v3/gate.json` | Restated G4 check 4 (`rollout_gate_v3.py`): student vs restricted "
     "da_gpsr, paired non-inferiority, 1 pp margin. |", 1),
    ("Carries the final `oracle_assignment`. |",
     "Carries the final `oracle_assignment`. Ran at altitude **50–150 m** (BASE). Evaluation "
     "reference only — the Dataset V3 label is da_gpsr everywhere. |", 1),
]

SPECS = [
    ('docs/DATASET_V2_MASTER_SPEC.md', 'SUPERSEDED (2026-10-02)', V2_SPEC),
    ('docs/FILE2_PRE_M5_EXECUTION_PLAN.md', 'PARTLY SUPERSEDED (2026-10-02)', FILE2),
    ('docs/FILE1_APPROACH2_RECORD.md', 'Corrections (2026-10-02)', FILE1),
    ('docs/CLAUDE.md', 'As of v26 (2026-10-02)', CLAUDE),
    ('docs/FANET_Full_Project_Report.md', 'DECIDED 2026-10-01: single global da_gpsr', REPORT),
    ('docs/Panel_Results_Report.md', '**50–150 m** — config_v2.BASE', PANEL),
    ('results/README.md', 'band_sink50_1000s_z100_300.json', RESULTS),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(a.root, 'docs', 'DATASET_V3_SPEC.md')):
        print('  docs/DATASET_V3_SPEC.md missing -- copy the v26 files in first.  NO FILE WRITTEN.')
        return 1
    staged, ok = {}, True
    for rel, guard, edits in SPECS:
        p = os.path.join(a.root, rel)
        if not os.path.isfile(p):
            print(f'  {rel:<38} MISSING  <-- ABORT'); ok = False
            continue
        text = io.open(p, encoding='utf-8').read()
        if guard in text:
            print(f'  {rel:<38} already applied -- skipped')
            continue
        new = text
        for i, (old, rep, want) in enumerate(edits, 1):
            n = new.count(old)
            if n != want:
                print(f'  {rel:<38} edit {i}: anchor matched {n}x, expected {want}  <-- ABORT')
                ok = False
                break
            new = new.replace(old, rep)
        else:
            staged[p] = new
            print(f'  {rel:<38} {len(edits)} edit(s) staged')
    if not ok:
        print('\n  NO FILE WRITTEN.')
        return 1
    if not staged:
        print('\n  ALREADY APPLIED.')
        return 0
    if a.dry_run:
        print(f'\n  DRY RUN OK -- {len(staged)} file(s) would be written.')
        return 0
    for p, t in staged.items():
        with io.open(p, 'w', encoding='utf-8', newline='') as f:
            f.write(t)
    print(f'\n  APPLIED to {len(staged)} file(s).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
