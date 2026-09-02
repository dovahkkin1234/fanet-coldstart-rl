"""
apply_rollout_mask_v7.py — PRIORITY 1. Feature masking in the rollout path.

WHY THIS IS BLOCKING
--------------------
train_supervised_v2.py can mask features; rollout_eval_v2.py cannot. Two
consequences, both fatal for the decentralised decision:

  1. There is NO PDR evidence for the masked models. The M4 masked result
     (GNN +0.0399, 30/30 seeds) is ACCURACY ONLY. We do not know what a
     masked policy actually delivers when it is put in charge of the network.

  2. M5's environment would feed the policy features it never trained on.
     A policy trained with hop_distance_to_dst zeroed, deployed in an
     environment that supplies the true value, is being handed a column whose
     meaning it never learned. It would not error -- it would just route
     worse, silently. Exactly the "plausible number on a dead mechanism"
     signature this project keeps catching.

WHAT IT DOES
------------
  M1  ModelActorSimulator gains a `mask` argument. Masking is applied in BOTH
      places the model reads features:
        - _encode()          -> node and edge feature blocks
        - _select_next_hop() -> query and candidate blocks
      Masking the query block alone would leave the candidate block live, so
      the ablation would be half-applied -- the same trap the training-side
      patch had to avoid with cand_edge.

  M2  --mask CLI accepting the same preset names as the trainer
      ('hop', 'gnnjob', 'none') plus raw feature names. Presets are IMPORTED
      from train_supervised_v2 rather than redefined, so the two paths cannot
      drift apart. A drifted mask would silently compare a policy against a
      differently-masked rollout.

  M3  STARTUP ASSERTION. Before any episode runs, build one real frame and one
      real decision and assert every masked column is exactly zero -- in all
      four blocks. A typo in a feature name would otherwise mask nothing and
      produce a "no difference" result that looks like a finding.

  M4  Results and checkpoints are TAGGED with the mask. rollout_masked-*.json
      cannot be merged with an unmasked rollout.json, so a masked policy can
      never be compared against unmasked reference numbers by accident.

WHAT TO EXPECT
--------------
Masked rollout PDR SHOULD BE LOWER than the unmasked 97.8% / 98.9%. Masked
top-1 accuracy is ~0.756 against ~0.912 unmasked. That drop is the measured
price of decentralisation and is to be reported, not hidden.

USAGE
    python apply_rollout_mask_v7.py --src src [--dry-run]
"""

import argparse
import os
import sys

VERSION = "v7"


class PatchError(RuntimeError):
    pass


def sub(text, old, new, label, path):
    n = text.count(old)
    if n != 1:
        raise PatchError(
            f"[{path}] anchor for '{label}' matched {n} times, expected 1.\n"
            f"  anchor starts: {old[:110]!r}")
    return text.replace(old, new, 1)


# ── M2: import the presets rather than redefining them ──────────────────────
A_OLD = """from train_supervised_v2 import (PhaseB, train_one, SEARCH_SPACE, MODELS,
                                 paired_stats)"""
A_NEW = """from train_supervised_v2 import (PhaseB, train_one, SEARCH_SPACE, MODELS,
                                 paired_stats, MASK_PRESETS, resolve_mask)"""


# ── M1: ModelActorSimulator takes and applies a mask ────────────────────────
B_OLD = """    def __init__(self, config, model, device):
        super().__init__(config)
        self.model = model
        self.device = device
        self.nc = F.norm_constants(config)"""

B_NEW = '''    def __init__(self, config, model, device, mask=None):
        super().__init__(config)
        self.model = model
        self.device = device
        self.nc = F.norm_constants(config)
        # Resolved ONCE, at construction: {block: [column indices]}. resolve_mask
        # raises on an unknown feature name, so a typo fails here rather than
        # silently masking nothing.
        self.mask = resolve_mask(mask or [])
        self.mask_names = list(mask or [])'''

C_OLD = """        _ids, nf, ei, ef = F.extract_frame(G, self.nc)"""
C_NEW = """        _ids, nf, ei, ef = F.extract_frame(G, self.nc)
        # Node and edge blocks. Applied to the numpy arrays BEFORE densify, so
        # the masked edge values propagate into both `dense` and the
        # cand_edge slice gathered out of it in _select_next_hop.
        for _c in self.mask.get('node', []):
            nf[:, _c] = 0.0
        for _c in self.mask.get('edge', []):
            ef[:, _c] = 0.0"""

D_OLD = """        qf, cf = F.extract_decision(G, pkt, cands, self.nc, h_map, 0.0, 0.0,
                                    ttl_const=TTL)"""
D_NEW = """        qf, cf = F.extract_decision(G, pkt, cands, self.nc, h_map, 0.0, 0.0,
                                    ttl_const=TTL)
        # Query and candidate blocks. Masking the query block alone would leave
        # the candidate block live and the ablation would be half-applied.
        for _c in self.mask.get('query', []):
            qf[_c] = 0.0
        for _c in self.mask.get('cand', []):
            cf[:, _c] = 0.0"""


# ── run_episode passes the mask through ─────────────────────────────────────
E_OLD = """def run_episode(cfg, scen_cfg, rate, seed, actor, model=None, device='cpu'):
    config = {**BASE, **scen_cfg, 'packet_rate': rate, 'seed': seed}
    if model is None:
        config['actor'] = actor
        return FANETSimulatorV2(config).run()
    sim = ModelActorSimulator(config, model, device)"""

E_NEW = """def run_episode(cfg, scen_cfg, rate, seed, actor, model=None, device='cpu',
                mask=None):
    config = {**BASE, **scen_cfg, 'packet_rate': rate, 'seed': seed}
    if model is None:
        config['actor'] = actor
        return FANETSimulatorV2(config).run()
    sim = ModelActorSimulator(config, model, device, mask=mask)"""


# ── M3: startup assertion ───────────────────────────────────────────────────
F_OLD = """def episode_grid(seeds, heldout='medium_slow'):"""

F_NEW = '''def assert_mask_applied(mask_names, device='cpu'):
    """Build one real frame and one real decision; assert every masked column
    is exactly zero in all four blocks.

    A mistyped feature name would otherwise mask nothing, and the run would
    produce a 'masking made no difference' result that looks like a finding.
    resolve_mask already rejects unknown names, so this checks the second
    failure mode: a name that resolves but is never actually zeroed because the
    write site was missed.
    """
    if not mask_names:
        return
    import numpy as _np
    scen, cfg = next(iter(SCENARIOS.items()))
    sim = ModelActorSimulator({**BASE, **cfg, 'packet_rate': 2.0, 'seed': 999},
                              model=None, device=device, mask=mask_names)
    G = sim._build_graph()
    _ids, nf, ei, ef = F.extract_frame(G, sim.nc)
    for _c in sim.mask.get('node', []):
        nf[:, _c] = 0.0
    for _c in sim.mask.get('edge', []):
        ef[:, _c] = 0.0
    bad = []
    for blk, arr in (('node', nf), ('edge', ef)):
        for c in sim.mask.get(blk, []):
            if float(abs(arr[:, c]).max()) != 0.0:
                bad.append(f'{blk}[{c}]')

    nodes = sorted(G.nodes())
    cur, dst = nodes[0], nodes[-1]
    h_map = F.hop_distances_to(G, dst)
    cands = canonical_candidates(G, cur, dst, set())
    if cands:
        class _P:
            pass
        p = _P(); p.current, p.dst, p.hops, p.path = cur, dst, 0, [cur]
        qf, cf = F.extract_decision(G, p, cands, sim.nc, h_map, 0.0, 0.0,
                                    ttl_const=TTL)
        for _c in sim.mask.get('query', []):
            qf[_c] = 0.0
        for _c in sim.mask.get('cand', []):
            cf[:, _c] = 0.0
        for c in sim.mask.get('query', []):
            if float(abs(qf[c])) != 0.0:
                bad.append(f'query[{c}]')
        for c in sim.mask.get('cand', []):
            if float(abs(cf[:, c]).max()) != 0.0:
                bad.append(f'cand[{c}]')
    if bad:
        raise AssertionError(
            f'MASK NOT APPLIED in the rollout path: {bad}. The rollout would '
            f'feed the policy features it was never trained on.')
    print(f'  masked (zeroed) in rollout: {mask_names}')
    print(f'    resolved to columns: {sim.mask}')
    print(f'    VERIFIED zero in a real frame and a real decision '
          f'(node, edge, query, candidate)')


def episode_grid(seeds, heldout='medium_slow'):'''


# ── M2/M4: CLI, tag, and threading the mask through main() ──────────────────
G_OLD = """    ap.add_argument('--repro', action='store_true',"""
G_NEW = """    ap.add_argument('--mask', nargs='+', default=None,
                    help="feature names to ZERO, or a preset: 'hop' or "
                         "'gnnjob'. MUST match the mask the checkpoint was "
                         "trained under, or the policy sees columns it never "
                         "learned to use.")
    ap.add_argument('--repro', action='store_true',"""

H_OLD = """    HP = dict(SEARCH_SPACE)"""
H_NEW = """    mask_names = []
    for m in (args.mask or []):
        mask_names += MASK_PRESETS.get(m, [m])

    HP = dict(SEARCH_SPACE)"""

I_OLD = """    _sfx = "".join(f"_{k}{HP[k]:g}" for k in sorted(_diff))
    resfile = os.path.join(args.out, f'rollout{_sfx}.json')"""
I_NEW = """    _sfx = "".join(f"_{k}{HP[k]:g}" for k in sorted(_diff))
    # The mask goes in the filename too. A masked rollout must never merge into
    # an unmasked rollout.json -- the two measure different policies.
    if mask_names:
        _sfx += "_masked-" + "+".join(sorted(n[:12] for n in mask_names))
    resfile = os.path.join(args.out, f'rollout{_sfx}.json')"""

J_OLD = """    ds = PhaseB(args.data)"""
J_NEW = """    ds = PhaseB(args.data, mask=mask_names)
    assert_mask_applied(mask_names, args.device)"""

K_OLD = """            model, val_best, eps = train_one(ds, mixer, seed,
                                             args.device, hp=HP)"""
K_NEW = """            model, val_best, eps = train_one(ds, mixer, seed,
                                             args.device, hp=HP)"""

# every run_episode call that drives the MODEL must pass the mask
L_OLD = """            for scen, cfg, rate, s, held in grid:
                m = run_episode(cfg, cfg, rate, s, None, model, args.device)
                pdr[f'{scen}|{rate}|{s}'] = float(m['network_pdr'])
            res[key] = pdr"""
L_NEW = """            for scen, cfg, rate, s, held in grid:
                m = run_episode(cfg, cfg, rate, s, None, model, args.device,
                                mask=mask_names)
                pdr[f'{scen}|{rate}|{s}'] = float(m['network_pdr'])
            res[key] = pdr"""

M_OLD = """            for scen, cfg, rate, s, held in grid:
                m = run_episode(cfg, cfg, rate, s, None, model, args.device)
                got[f'{scen}|{rate}|{s}'] = float(m['network_pdr'])"""
M_NEW = """            for scen, cfg, rate, s, held in grid:
                m = run_episode(cfg, cfg, rate, s, None, model, args.device,
                                mask=mask_names)
                got[f'{scen}|{rate}|{s}'] = float(m['network_pdr'])"""

N_OLD = """            torch.save({'state_dict': model.state_dict(), 'mixer': mixer,
                        'seed': seed, 'val_contested': val_best,
                        'schema': F.FEATURE_SCHEMA_VERSION,
                        'hp': HP}, ckpt)"""
N_NEW = """            torch.save({'state_dict': model.state_dict(), 'mixer': mixer,
                        'seed': seed, 'val_contested': val_best,
                        'schema': F.FEATURE_SCHEMA_VERSION,
                        'mask': mask_names,      # M5 must load with this mask
                        'hp': HP}, ckpt)"""

O_OLD = """    print(f'  shared budget (M-14): {HP}')"""
O_NEW = """    print(f'  shared budget (M-14): {HP}')
    if mask_names:
        print(f'  MASKED ROLLOUT -- expect PDR BELOW the unmasked 97.8%/98.9%.')
        print(f'  That drop is the measured price of decentralisation.')"""

P_OLD = """from generate_dataset_v2 import (SCENARIOS, RATES, BASE, TTL,
                                 canonical_candidates)"""
P_NEW = """from generate_dataset_v2 import (SCENARIOS, RATES, BASE, TTL,
                                 canonical_candidates)"""


PATCHES = {
    'rollout_eval_v2.py': [
        (A_OLD, A_NEW, 'M2 import presets from the trainer'),
        (B_OLD, B_NEW, 'M1 ModelActorSimulator takes a mask'),
        (C_OLD, C_NEW, 'M1 mask node + edge blocks in _encode'),
        (D_OLD, D_NEW, 'M1 mask query + candidate blocks in _select_next_hop'),
        (E_OLD, E_NEW, 'M1 run_episode threads the mask'),
        (F_OLD, F_NEW, 'M3 startup assertion'),
        (G_OLD, G_NEW, 'M2 --mask CLI'),
        (H_OLD, H_NEW, 'M2 preset expansion'),
        (I_OLD, I_NEW, 'M4 mask in the results filename'),
        (J_OLD, J_NEW, 'M2/M3 dataset mask + assertion call'),
        (L_OLD, L_NEW, 'M1 main rollout loop passes the mask'),
        (M_OLD, M_NEW, 'M1 repro rollout loop passes the mask'),
        (N_OLD, N_NEW, 'M4 checkpoint records its mask'),
        (O_OLD, O_NEW, 'M4 expectation banner'),
    ],
}

MARKER = 'assert_mask_applied'
GUARD = 'rollout_eval_v2.py'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    g = os.path.join(args.src, GUARD)
    if not os.path.isfile(g):
        print(f"ERROR: {g} not found. Run from the repo root or pass --src.")
        return 2
    txt = open(g, encoding='utf-8').read()
    if MARKER in txt:
        print(f"ERROR: {GUARD} already patched by v7. Restore from git first.")
        return 2
    if '_frame_no' not in txt:
        print(f"ERROR: {GUARD} lacks the frame-counter cache fix.\n"
              f"  v7 applies on top of it. Pull the latest main first.")
        return 2

    staged = {}
    print(f"\n{'=' * 78}\n  ROLLOUT MASK SUPPORT {VERSION} — assertion-guarded\n{'=' * 78}")
    for fname, edits in PATCHES.items():
        path = os.path.join(args.src, fname)
        text = open(path, encoding='utf-8').read()
        for old, new, label in edits:
            if old == new:
                continue
            text = sub(text, old, new, label, fname)
            print(f"    [ok] {fname:<24} {label}")
        staged[path] = text

    if args.dry_run:
        print(f"\n  DRY RUN — {len(staged)} file(s) would change, nothing written.")
        return 0
    for path, text in staged.items():
        open(path, 'w', encoding='utf-8').write(text)
    print(f"\n  WROTE {len(staged)} file(s).\n")
    print("  NEXT — the masked check-4 run (PRIORITY 1):")
    print("    python src/rollout_eval_v2.py --data data/phaseB \\")
    print("      --out results/m5_masked --seeds 3 --mask hop \\")
    print("      --lr 1e-3 --attn_dropout 0.1 --max_epochs 100 \\")
    print("      --mixers attention mlp")
    print("\n  Compare its PDR against the unmasked 97.8% (attention) /")
    print("  98.9% (mlp). The gap IS the price of decentralisation.")
    print("=" * 78 + "\n")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except PatchError as e:
        print(f"\nPATCH FAILED — nothing was written.\n{e}\n")
        sys.exit(1)
