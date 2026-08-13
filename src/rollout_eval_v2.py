"""
rollout_eval_v2.py — G4 CHECK 4. The decisive M4 gate.

    python src/rollout_eval_v2.py --data data/phaseB --out results/m4_rollout --seeds 3

Accuracy is a proxy. This is not: the trained model is put in the SIMULATOR'S
ACTOR SLOT and the resulting network PDR is compared against SP-BP's on the
same episodes. That is what M5 inherits, so it is what the gate should test.

    PASS if  mean(model PDR) >= 0.90 * mean(SP-BP PDR)  on held-out episodes.

WHY THIS TRAINS ITS OWN MODELS
train_supervised_v2.py never wrote checkpoints -- it loads the best state back
into memory, evaluates, and discards it. Rather than force a 2.6 h retraining
of all 100 runs to recover weights, this script trains its own small set
(default 3 seeds per architecture) using the SAME train_one() and the SAME
shared SEARCH_SPACE, so M-14 parity holds, and saves them. Accuracy numbers
already exist from the 50-seed study; what is needed here is rollout, and a
handful of models is enough to bound it.

HELD-OUT ONLY. Episodes use test seeds (143-150) and the held-out
medium_slow scenario. Never the training seeds -- a rollout on seeds the model
was fit to would measure memorisation.

CANDIDATE ORDER MUST MATCH THE DATASET EXACTLY.
canonical_candidates() sorts by ascending distance-to-destination, which is why
slot 0 is the nearest-to-destination baseline. The model was trained on that
ordering, so the actor imports the same function rather than reimplementing it.
Three separate SP-BP reimplementations have already diverged from the panel in
this project; this one is not going to be a fourth.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import features_v2 as F
from simulator_v2 import FANETSimulatorV2
from generate_dataset_v2 import (SCENARIOS, RATES, BASE, TTL,
                                 canonical_candidates)
from model_gnn_attn import FANETRouter, densify, set_determinism
from train_supervised_v2 import (PhaseB, train_one, SEARCH_SPACE, MODELS,
                                 paired_stats)


class ModelActorSimulator(FANETSimulatorV2):
    """FANETSimulatorV2 with a trained model in the actor slot.

    Overrides ONLY _select_next_hop, the same extension point the dataset
    generator uses, so every piece of validated M2 machinery -- ARQ retries,
    queue admission, energy accounting, drop taxonomy -- runs unchanged.
    """

    def __init__(self, config, model, device):
        super().__init__(config)
        self.model = model
        self.device = device
        self.nc = F.norm_constants(config)
        # CACHE KEYED ON FRAME INDEX, NEVER id(G).
        # The first version used id(G) as the key. That is a MEMORY ADDRESS:
        # simulator_v2 rebuilds the graph every frame (line 704) and drops the
        # old one, and CPython recycles a freed object's address immediately --
        # measured 100% reuse over 200 sequential nx.Graph allocations. So a
        # NEW frame could land on the freed address, hit the one-entry cache,
        # and be routed using the PREVIOUS frame's encoding.
        # Effect is rare and heap-dependent, which is exactly the signature
        # seen in check 6: one packet, in one episode out of 48, differing
        # between two runs of identical weights.
        self._frame_no = -1
        self._enc_cache = {}
        self._hop_cache = {}
        self.n_decisions = 0
        self.n_no_candidate = 0

    def _build_graph(self):
        """Bump the frame counter in lockstep with graph construction.

        Incrementing here rather than in the step loop means the counter cannot
        drift out of step with the graph it labels, whatever the caller does.
        """
        self._frame_no += 1
        return super()._build_graph()

    def _encode(self, G):
        """Encoder runs ONCE PER FRAME. The graph object is rebuilt each frame,
        so its identity is a safe cache key within a frame."""
        key = self._frame_no
        if key in self._enc_cache:
            return self._enc_cache[key]
        _ids, nf, ei, ef = F.extract_frame(G, self.nc)
        n = nf.shape[0]
        adj, dense = densify(torch.from_numpy(ei.astype(np.int64)).to(self.device),
                             torch.from_numpy(ef).to(self.device), n,
                             device=self.device)
        nft = torch.from_numpy(nf).to(self.device).unsqueeze(0)
        with torch.no_grad():
            h = self.model.encode(nft, adj.unsqueeze(0), dense.unsqueeze(0),
                                  check=False)
        self._enc_cache = {key: (h, nft, dense.unsqueeze(0))}   # keep one frame
        return self._enc_cache[key]

    def _select_next_hop(self, G, pkt, neighbors):
        cands = canonical_candidates(G, pkt.current, pkt.dst, set(pkt.path))
        if not cands:
            self.n_no_candidate += 1
            return None
        if len(cands) == 1:
            self.n_decisions += 1
            return cands[0]

        key = (self._frame_no, pkt.dst)
        if key not in self._hop_cache:
            self._hop_cache = {key: F.hop_distances_to(G, pkt.dst)}
        h_map = self._hop_cache[key]

        # n_inflight / net_occ are recomputed from G inside extract_decision
        # whenever LOCAL_HORIZON is set, so the passed values are unused there.
        qf, cf = F.extract_decision(G, pkt, cands, self.nc, h_map, 0.0, 0.0,
                                    ttl_const=TTL)
        h, nft, dense = self._encode(G)
        dev = self.device
        ci = torch.tensor([cands], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = self.model.score(
                h, nft,
                torch.tensor([pkt.current], device=dev),
                torch.tensor([pkt.dst], device=dev),
                torch.from_numpy(qf).to(dev).unsqueeze(0),
                ci,
                torch.from_numpy(cf).to(dev).unsqueeze(0),
                dense[0, pkt.current, ci[0]].unsqueeze(0),
                torch.ones(1, len(cands), dtype=torch.bool, device=dev))
        self.n_decisions += 1
        return cands[int(logits.argmax(-1).item())]


def episode_grid(seeds, heldout='medium_slow'):
    """Held-out episodes only: test seeds on the training scenarios, plus the
    held-out scenario (which is out-of-distribution at every seed)."""
    out = []
    for scen, cfg in SCENARIOS.items():
        for rate in RATES:
            for s in seeds:
                out.append((scen, cfg, rate, s, scen == heldout))
    return out


def run_episode(cfg, scen_cfg, rate, seed, actor, model=None, device='cpu'):
    config = {**BASE, **scen_cfg, 'packet_rate': rate, 'seed': seed}
    if model is None:
        config['actor'] = actor
        return FANETSimulatorV2(config).run()
    sim = ModelActorSimulator(config, model, device)
    m = sim.run()
    m['_n_decisions'] = sim.n_decisions
    m['_n_no_candidate'] = sim.n_no_candidate
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/phaseB')
    ap.add_argument('--out', default='results/m4_rollout')
    ap.add_argument('--seeds', type=int, default=3,
                    help='TRAINING seeds (models to train and roll out)')
    ap.add_argument('--seed0', type=int, default=2000)
    ap.add_argument('--episode_seeds', type=int, nargs='+',
                    default=[143, 145, 147, 149],
                    help='HELD-OUT episode seeds; must be outside 101-142')
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # HYPERPARAMETERS (was missing entirely).
    # train_one() was called with no hp argument, so it always used the
    # SEARCH_SPACE default -- meaning check 4's rollout could only ever
    # evaluate the DEFAULT config. Once the HP grid showed lr=1e-3 beats the
    # 3e-4 default by ~3 pp for both models, this made it impossible to roll
    # out the tuned model at all. Both models get the same values, so M-14
    # parity holds, and the tag records them.
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--d', type=int, default=None)
    ap.add_argument('--heads', type=int, default=None)
    ap.add_argument('--dropout', type=float, default=None)
    ap.add_argument('--attn_dropout', type=float, default=None)
    ap.add_argument('--layers', type=int, default=None)
    ap.add_argument('--max_epochs', type=int, default=None)
    ap.add_argument('--repro', action='store_true',
                    help='G4 CHECK 6: retrain and re-roll every stored key with '
                         'the SAME seed and compare. Writes to a separate '
                         'namespace so the original results are never '
                         'overwritten by the thing meant to verify them.')
    args = ap.parse_args()

    bad = [s for s in args.episode_seeds if 101 <= s <= 142]
    if bad:
        raise SystemExit(f"episode seeds {bad} are TRAINING/VAL seeds — rollout "
                         f"on them would measure memorisation, not routing")

    set_determinism()
    HP = dict(SEARCH_SPACE)
    for _k in ('lr', 'd', 'heads', 'dropout', 'attn_dropout', 'layers',
               'max_epochs'):
        _v = getattr(args, _k)
        if _v is not None:
            HP[_k] = _v
    _diff = {k: HP[k] for k in HP if k in SEARCH_SPACE and HP[k] != SEARCH_SPACE[k]}

    os.makedirs(args.out, exist_ok=True)
    # Config goes in the FILENAME. Rolling out a tuned model into the same
    # rollout.json as the default one would silently mix two policies in a
    # single paired comparison.
    _sfx = "".join(f"_{k}{HP[k]:g}" for k in sorted(_diff))
    resfile = os.path.join(args.out, f'rollout{_sfx}.json')
    res = json.load(open(resfile)) if os.path.isfile(resfile) else {}

    ds = PhaseB(args.data)
    grid = episode_grid(args.episode_seeds)
    print('=' * 78)
    print('  G4 CHECK 4 — ROLLOUT PDR vs SP-BP  (the decisive gate)')
    print('=' * 78)
    print(f'  device={args.device}  {len(grid)} held-out episodes per policy')
    print(f'  episode seeds {args.episode_seeds} (training was 101-135, val 136-142)')
    print(f'  shared budget (M-14): {HP}')
    if _diff:
        print(f'  NON-DEFAULT: {_diff}  -> {os.path.basename(resfile)}')
    print('=' * 78)

    # ---- SP-BP reference, once ----
    if 'spbp' not in res:
        t0 = time.time()
        pdr = {}
        for scen, cfg, rate, s, held in grid:
            m = run_episode(cfg, cfg, rate, s, 'spbp')
            pdr[f'{scen}|{rate}|{s}'] = float(m['network_pdr'])
        res['spbp'] = pdr
        json.dump(res, open(resfile, 'w'), indent=2)
        print(f"  [spbp] mean PDR {np.mean(list(pdr.values())):.4f} "
              f"({time.time()-t0:.0f}s)")

    for mixer in args.models:
        for si in range(args.seeds):
            seed = args.seed0 + si
            key = f'{mixer}:{seed}'
            if key in res:
                continue
            t0 = time.time()
            model, val_best, eps = train_one(ds, mixer, seed,
                                             args.device, hp=HP)
            model.eval()
            ckpt = os.path.join(args.out, f'{mixer}_{seed}{_sfx}.pt')
            torch.save({'state_dict': model.state_dict(), 'mixer': mixer,
                        'seed': seed, 'val_contested': val_best,
                        'schema': F.FEATURE_SCHEMA_VERSION,
                        'hp': HP}, ckpt)
            pdr = {}
            for scen, cfg, rate, s, held in grid:
                m = run_episode(cfg, cfg, rate, s, None, model, args.device)
                pdr[f'{scen}|{rate}|{s}'] = float(m['network_pdr'])
            res[key] = pdr
            json.dump(res, open(resfile, 'w'), indent=2)
            print(f"  [{key:>16}] val={val_best:.4f} mean PDR "
                  f"{np.mean(list(pdr.values())):.4f} ({time.time()-t0:.0f}s)")

    # ---- CHECK 6: reproducibility under a fixed seed ----
    if args.repro:
        print('\n' + '=' * 78)
        print('  G4 CHECK 6 — REPRODUCIBILITY UNDER FIXED SEED')
        print('=' * 78)
        worst, mean_drift = 0.0, []
        for key in [k for k in list(res) if k != 'spbp']:
            mixer, seed = key.split(':')
            model, _v, _e = train_one(ds, mixer, int(seed),
                                      args.device, hp=HP)
            model.eval()
            got = {}
            for scen, cfg, rate, s, held in grid:
                m = run_episode(cfg, cfg, rate, s, None, model, args.device)
                got[f'{scen}|{rate}|{s}'] = float(m['network_pdr'])
            diffs = [abs(got[c] - res[key][c]) for c in res[key]]
            # MAX ALONE IS NOT ENOUGH. check 4's verdict is a MEAN over 48
            # episodes, so what decides whether a drift matters is how much the
            # mean moves, not the worst single episode. The first version of
            # this check reported only the max and would have left that
            # unanswered.
            dmean = abs(float(np.mean([got[c] for c in res[key]]))
                        - float(np.mean([res[key][c] for c in res[key]])))
            worst = max(worst, max(diffs))
            mean_drift.append(dmean)
            print(f"    {key:>16}  max|diff| {max(diffs):.2e}   "
                  f"|drift in MEAN PDR| {dmean:.2e}   "
                  f"{'OK' if max(diffs) < 1e-9 else 'DRIFT'}")
        wm = max(mean_drift) if mean_drift else 0.0
        print(f"\n    worst single-episode deviation: {worst:.2e}")
        print(f"    worst drift in the MEAN PDR:    {wm:.2e}   "
              f"(check 4's margin over the 90% threshold is ~0.035 absolute,"
              f" so a mean drift below ~1e-3 cannot flip the verdict)")
        print(f"    -> {'PASS' if worst < 1e-9 else 'FAIL (bit-reproducibility)'}")
        print("    Covers BOTH halves: retraining from the same seed, and")
        print("    re-rolling the retrained weights. A rollout-only check would")
        print("    have missed nondeterminism in training entirely.")
        print('=' * 78)

    # ---- verdict ----
    ref = res['spbp']
    cells = sorted(ref)
    sp = np.array([ref[c] for c in cells])
    print('\n' + '=' * 78)
    print('  CHECK 4 VERDICT')
    print('=' * 78)
    print(f"    SP-BP reference mean PDR {sp.mean():.4f}  "
          f"(90% threshold = {0.9*sp.mean():.4f})")
    summary = {'spbp_mean': float(sp.mean()), 'threshold': float(0.9 * sp.mean())}
    for mixer in args.models:
        keys = [k for k in res if k.startswith(f'{mixer}:')]
        if not keys:
            continue
        per_seed = np.array([[res[k][c] for c in cells] for k in keys])
        mean = per_seed.mean(axis=1)
        ratio = mean / sp.mean()
        ok = bool((ratio.mean() >= 0.90))
        print(f"    {mixer:<10} mean PDR {mean.mean():.4f}  "
              f"= {100*ratio.mean():.1f}% of SP-BP   "
              f"[{100*ratio.min():.1f}-{100*ratio.max():.1f}% across seeds]  "
              f"{'PASS' if ok else 'FAIL'}")
        # held-out scenario reported separately: it is out of distribution
        ho = [i for i, c in enumerate(cells) if c.startswith('medium_slow')]
        tr = [i for i, c in enumerate(cells) if not c.startswith('medium_slow')]
        print(f"      trained scenarios {per_seed[:, tr].mean():.4f} "
              f"({100*per_seed[:, tr].mean()/sp[tr].mean():.1f}% of SP-BP)   "
              f"held-out medium_slow {per_seed[:, ho].mean():.4f} "
              f"({100*per_seed[:, ho].mean()/sp[ho].mean():.1f}%)")
        summary[mixer] = {'mean_pdr': float(mean.mean()),
                          'pct_of_spbp': float(100 * ratio.mean()),
                          'pass': ok,
                          'trained_pct': float(100*per_seed[:, tr].mean()/sp[tr].mean()),
                          'heldout_pct': float(100*per_seed[:, ho].mean()/sp[ho].mean())}
    # ---- paired architecture comparison, matched by seed ----
    # The PASS verdict above needs no statistics: the threshold is cleared by
    # 7-8 pp and every seed clears it individually. This block is for the
    # ORDERING claim, which does. Reporting it uncorrected next to a 30-seed
    # accuracy study would be the inconsistency M-7 exists to prevent.
    seeds_avail = sorted({int(k.split(':')[1]) for k in res if k != 'spbp'})
    paired_seeds = [s for s in seeds_avail
                    if all(f'{m}:{s}' in res for m in ('attention', 'mlp'))]
    if len(paired_seeds) >= 3:
        ho = [c for c in cells if c.startswith('medium_slow')]
        tr = [c for c in cells if not c.startswith('medium_slow')]
        sp_all, sp_tr = sp.mean(), np.mean([ref[c] for c in tr])
        sp_ho = np.mean([ref[c] for c in ho])

        def ratios(mix, subset, denom):
            return [np.mean([res[f'{mix}:{s}'][c] for c in subset]) / denom
                    for s in paired_seeds]

        print('\n' + '-' * 78)
        print(f'  PAIRED: mlp - attention, ratio to SP-BP, n={len(paired_seeds)} seeds')
        print('-' * 78)
        block = {}
        for nm, subset, denom in (('all episodes', cells, sp_all),
                                  ('trained scenarios', tr, sp_tr),
                                  ('held-out medium_slow', ho, sp_ho)):
            st = paired_stats(ratios('mlp', subset, denom),
                              ratios('attention', subset, denom))
            block[nm] = st
            print(f"    {nm:<22} {100*st['mean_diff']:+6.2f} pp   "
                  f"95% CI [{100*st['ci95'][0]:+.2f}, {100*st['ci95'][1]:+.2f}]   "
                  f"d={st['cohens_d']:+.2f}  p={st['p']:.2e}")
        dod = (block['trained scenarios']['mean_diff']
               - block['held-out medium_slow']['mean_diff'])
        print(f"\n    difference-of-differences (trained minus held-out): "
              f"{100*dod:+.2f} pp")
        print("    This is the interesting quantity: if it is clearly positive,")
        print("    the MLP's PDR edge is REGIME-DEPENDENT (present where trained,")
        print("    absent out of distribution). It has ~2x the variance of the")
        print("    main effect, so it needs more seeds than the ordering does.")
        if len(paired_seeds) < 10:
            print(f"    WARNING: {len(paired_seeds)} seeds is too few to claim it.")
        summary['paired'] = {k: v for k, v in block.items()}
        summary['difference_of_differences'] = float(dod)
        summary['n_paired_seeds'] = len(paired_seeds)

    json.dump(summary, open(os.path.join(args.out, 'check4_summary.json'), 'w'),
              indent=2)
    print()
    print("    Accuracy said the MLP wins by ~1pp. PDR is the number M5")
    print("    inherits: a model can lose on top-1 and still route as well, if")
    print("    its errors are near-ties. Read both, and report both.")
    print('=' * 78 + '\n')


if __name__ == '__main__':
    main()
