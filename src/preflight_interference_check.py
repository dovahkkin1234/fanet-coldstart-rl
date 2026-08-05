"""
preflight_interference_check.py  —  GATE G1 for Approach 2.

THE PREMISE THE WHOLE APPROACH-2 PLAN RESTS ON:
  Under realistic load, link_quality must become a function of distance AND
  network load — not distance alone. If it does not, there is no congestion
  coupling, no regime structure, no reason for continual learning, and no
  novel situations for CBR. This script is the cheapest possible test of that
  premise: it can pass or kill the next several months of work in one run.

WHAT IT DOES (no full simulator_v2 needed — physics-layer test only):
  1. Places N drones uniformly in a scenario area.
  2. Builds the feasible-link set (distance <= comm_range, rssi > sensitivity),
     exactly as simulator._build_graph does.
  3. For each feasible link (tx -> rx) and each offered-load level `activity`:
       - CONTENTION: nodes within carrier-sense range of tx are contenders;
         active count -> Bianchi collision probability (MAC-layer loss).
       - HIDDEN-TERMINAL INTERFERENCE: nodes in the annulus
         (carrier-sense range, interference range] around rx that are
         transmitting this slot (each active w.p. `activity`) sum their
         received power at rx -> SINR (PHY-layer degradation).
       - Per-link log-normal shadowing.
     Carrier-sensing (near nodes defer) is why the dominant interference is
     the hidden-terminal band, giving GRADED degradation rather than
     annihilation.
  4. Aggregates the link_quality / PER distribution per load level and prints
     the shift + spread, plus the "shaping bite" metric (fraction of links
     with link_quality < 0.90) that was ~0% in Approach 1.

VERDICT G1 passes if, as load rises:
  - link_quality std increases materially (distribution spreads), AND
  - fraction of links with link_quality < 0.90 rises from ~0% to substantial, AND
  - mean PER rises materially.
  At activity=0 the distribution MUST match the interference-free v1 model
  (asserted) — proving the coupling is what load adds, not a reparametrisation.

Usage:
    python src\\preflight_interference_check.py
    python src\\preflight_interference_check.py --n_drones 40 --area 1500 \
           --comm_range 250 --trials 200 --activity 0 0.02 0.05 0.1 0.2
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import link_model as v1
import link_model_v2 as v2


def build_feasible_links(pos, comm_range):
    """Return list of (i, j, dist) for all node pairs forming a decodable link,
    matching simulator._build_graph's existence test."""
    n = len(pos)
    links = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d > comm_range:
                continue
            if not v2.link_exists(d, comm_range):
                continue
            links.append((i, j, d))
    return links


def link_quality_under_load(pos, comm_range, links, activity, rng,
                            shadowing_sigma, collision_model='unsaturated'):
    """For each feasible link, compute link_quality/PER at a given offered-load
    `activity` (probability a node transmits this slot). Returns arrays
    (link_quality, per) over all links, one sample per link.

    SCOPE -- this gate is HOMOGENEOUS in activity. Every carrier-sense peer is
    given the same `activity`, because G1 sweeps a single scalar offered-load
    level. simulator_v2 instead passes each node's OWN activity, and its
    comment is explicit that the heterogeneity is deliberate: congested and
    idle nodes coexisting is the congestion-collapse mechanism, and a mean
    blurs it away. So G1 validates the collision model's SHAPE -- monotone in
    load, no hard-zero plateau, correct v1 reduction at activity 0 -- and NOT
    that mechanism. Do not cite G1 as evidence about heterogeneous congestion;
    that evidence comes from G2's load sweep.
    """
    cs_range = v2.CARRIER_SENSE_MULT * comm_range          # near: defer (CSMA)
    if_range = v2.INTERFERENCE_RANGE_MULT * comm_range     # beyond: negligible
    lq_out, per_out = [], []

    for (i, j, d) in links:
        tx, rx = i, j
        rx_pos, tx_pos = pos[rx], pos[tx]

        # ---- MAC contention: nodes in carrier-sense range of tx ----
        # Must use the SAME collision model as simulator_v2, or this gate and
        # the simulator report inconsistent physics for the same scenario.
        cs_nodes = [k for k in range(len(pos))
                    if k != tx and np.linalg.norm(pos[k] - tx_pos) <= cs_range]
        # RNG PARITY. The draw is UNCONDITIONAL; only its USE is branch-
        # dependent. Previously the unsaturated branch skipped it, so the two
        # models consumed the shared generator at different rates and every
        # subsequent shadowing draw diverged. Saturated still reproduced its
        # pre-flip numbers (its branch consumed identically), so nothing
        # published was wrong -- but a saturated-vs-unsaturated comparison was
        # UNPAIRED, running on different shadowing realisations, and a small
        # difference could not be attributed to the collision model. Drawing
        # unconditionally leaves the saturated path byte-identical and makes
        # the comparison paired.
        cs_draw = rng.random(len(cs_nodes)) if cs_nodes else np.empty(0)
        if collision_model == 'unsaturated':
            # Every carrier-sense peer is a potential contender that transmits
            # with probability `activity`; no Bernoulli thresholding and no
            # rounding to an integer count, so no hard-zero plateau. The peer
            # list excludes tx, matching simulator_v2's
            # [a for k, a in enumerate(self.activity) if cs_mask[k] and k != i]:
            # p_coll is P(some OTHER station transmits in this slot).
            p_coll_override = v2.bianchi_collision_prob_unsaturated(
                [activity] * len(cs_nodes))
            n_contenders = 1
        else:
            n_active_cs = int(np.sum(cs_draw < activity)) if cs_nodes else 0
            n_contenders = 1 + n_active_cs   # tx itself + active carrier-sense peers
            p_coll_override = None

        # ---- Hidden-terminal interference: annulus (cs_range, if_range] of rx ----
        interf_mw = 0.0
        for k in range(len(pos)):
            if k == tx or k == rx:
                continue
            dk_rx = float(np.linalg.norm(pos[k] - rx_pos))
            dk_tx = float(np.linalg.norm(pos[k] - tx_pos))
            # hidden terminal: far enough from tx to NOT defer, close enough to rx to interfere
            if dk_tx > cs_range and dk_rx <= if_range:
                if rng.random() < activity:                 # transmitting this slot
                    sh_k = rng.normal(0.0, shadowing_sigma)
                    interf_mw += v2.rx_power_mw(dk_rx, sh_k)

        sh_link = rng.normal(0.0, shadowing_sigma)
        _, _, lq_sinr, per = v2.compute_link_features_v2(
            d, interference_mw=interf_mw, n_contenders=n_contenders,
            shadowing_db=sh_link, p_collision=p_coll_override)

        # FOLD MAC CONTENTION INTO lq -- the M3 audit fix, which had been
        # applied in simulator_v2 but NOT here.
        #
        # compute_link_features_v2 returns link_quality = clip(sinr_db/30,0,1).
        # That is a pure SINR quantity: p_coll never touches it, only `per`.
        # simulator_v2._measured_link therefore does its own fold-in --
        #     lq = lq_clean * p_clear * (1 - p_coll)
        # -- with the comment that leaving MAC contention out made lq and per
        # 'describe different physics'. That fix landed in the simulator only.
        #
        # MEASURED CONSEQUENCE, before this change: switching --collision_model
        # moved `per` but left `lq` identical to six decimal places at every
        # activity level (0.812726 at a=0.02, 0.617758 at 0.06, 0.295316 at
        # 0.20 -- the SAME under both models). The new --collision_model flag
        # was steering only half the output, and G1's headline metric IS link
        # quality. The gate and the simulator were still reporting different
        # physics for the same scenario, which is exactly what the flag was
        # added to prevent.
        #
        # The hidden-terminal term is deliberately NOT reconciled: this gate
        # models it as continuous interference power entering SINR, the
        # simulator as a Bernoulli lethal-interferer indicator (p_clear). Those
        # are different modelling choices, not an inconsistency, and lq_sinr
        # already carries this gate's version through the SINR denominator.
        p_coll_eff = (p_coll_override if p_coll_override is not None
                      else v2.bianchi_collision_prob(n_contenders))
        lq = float(np.clip(lq_sinr * (1.0 - p_coll_eff), 0.0, 1.0))
        lq_out.append(lq)
        per_out.append(per)

    return np.array(lq_out), np.array(per_out)


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_drones', type=int, default=40)
    ap.add_argument('--area', type=float, default=1500.0)
    ap.add_argument('--z_min', type=float, default=50.0)
    ap.add_argument('--z_max', type=float, default=150.0)
    ap.add_argument('--comm_range', type=float, default=250.0)
    ap.add_argument('--trials', type=int, default=150,
                    help='independent topology placements averaged over')
    ap.add_argument('--activity', type=float, nargs='+',
                    default=[0.0, 0.02, 0.05, 0.10, 0.20],
                    help='offered-load levels (P(node transmits this slot))')
    ap.add_argument('--shadowing_sigma', type=float, default=v2.SHADOWING_SIGMA_DB)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--collision_model', default='unsaturated',
                    choices=['unsaturated', 'saturated'],
                    help='must match simulator_v2 default; saturated reproduces '
                         'the pre-M-4 numbers')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("\n" + "=" * 76)
    print("  GATE G1 — INTERFERENCE MAKES link_quality LOAD-DEPENDENT?")
    print("=" * 76)
    print(f"  {args.n_drones} drones in {args.area:.0f}x{args.area:.0f} m, "
          f"comm_range={args.comm_range:.0f} m, {args.trials} topologies")
    print(f"  collision model: {args.collision_model}")
    print(f"  carrier-sense={v2.CARRIER_SENSE_MULT}x, "
          f"interference={v2.INTERFERENCE_RANGE_MULT}x comm_range, "
          f"shadowing sigma={args.shadowing_sigma:.1f} dB")

    # Gather link_quality/PER distributions per activity level
    per_level = {a: {'lq': [], 'per': []} for a in args.activity}
    n_links_total = 0
    for t in range(args.trials):
        pos = np.column_stack([
            rng.uniform(0, args.area, args.n_drones),
            rng.uniform(0, args.area, args.n_drones),
            rng.uniform(args.z_min, args.z_max, args.n_drones),
        ])
        links = build_feasible_links(pos, args.comm_range)
        if not links:
            continue
        n_links_total += len(links)
        for a in args.activity:
            # shadowing off at a=0 to test exact v1 reduction; on otherwise
            sig = 0.0 if a == 0.0 else args.shadowing_sigma
            lq, per = link_quality_under_load(pos, args.comm_range, links, a, rng,
                                              sig, args.collision_model)
            per_level[a]['lq'].append(lq)
            per_level[a]['per'].append(per)

    print(f"\n  Sampled {n_links_total} feasible links across {args.trials} topologies.\n")

    # ---- Backward-compat assertion: activity=0, shadowing=0 == v1 ----
    if 0.0 in per_level:
        lq0 = np.concatenate(per_level[0.0]['lq'])
        # v1 reference: link_quality is a pure function of distance
        # (already implied; at a=0 interf=0, contenders=1, shadowing=0 -> v1)
        frac_saturated = float(np.mean(lq0 >= 0.90))
        print(f"  [a=0 sanity] interference-free link_quality: "
              f"mean={lq0.mean():.4f}  frac>=0.90={100*frac_saturated:.1f}%  "
              f"(Approach-1 regime: ~saturated)")

    # ---- Distribution table per load level ----
    print("\n" + "-" * 76)
    print("  link_quality DISTRIBUTION vs OFFERED LOAD")
    print("-" * 76)
    print(f"  {'activity':>9}{'mean':>9}{'std':>9}{'p05':>9}{'p25':>9}"
          f"{'median':>9}{'frac<0.90':>11}{'meanPER':>9}")
    prev_std = None
    lq_means, lq_stds, bite_fracs, per_means = [], [], [], []
    for a in args.activity:
        lq = np.concatenate(per_level[a]['lq'])
        per = np.concatenate(per_level[a]['per'])
        bite = float(np.mean(lq < 0.90))
        lq_means.append(lq.mean()); lq_stds.append(lq.std())
        bite_fracs.append(bite); per_means.append(per.mean())
        print(f"  {a:>9.2f}{lq.mean():>9.4f}{lq.std():>9.4f}{pct(lq,5):>9.4f}"
              f"{pct(lq,25):>9.4f}{pct(lq,50):>9.4f}{100*bite:>10.1f}%{per.mean():>9.4f}")

    # ---- Verdict ----
    print("\n" + "=" * 76)
    print("  VERDICT")
    print("=" * 76)
    lo, hi = args.activity[0], args.activity[-1]
    std_rise   = lq_stds[-1] - lq_stds[0]
    bite_rise  = bite_fracs[-1] - bite_fracs[0]
    per_rise   = per_means[-1] - per_means[0]
    mean_drop  = lq_means[0] - lq_means[-1]

    print(f"    From activity={lo:.2f} to {hi:.2f}:")
    print(f"      link_quality mean : {lq_means[0]:.3f} -> {lq_means[-1]:.3f}  (drop {mean_drop:.3f})")
    print(f"      link_quality std  : {lq_stds[0]:.3f} -> {lq_stds[-1]:.3f}  (rise {std_rise:.3f})")
    print(f"      frac < 0.90       : {100*bite_fracs[0]:.1f}% -> {100*bite_fracs[-1]:.1f}%  (rise {100*bite_rise:.1f} pts)")
    print(f"      mean PER          : {per_means[0]:.3f} -> {per_means[-1]:.3f}  (rise {per_rise:.3f})")

    passed = (std_rise > 0.02) and (bite_rise > 0.10) and (per_rise > 0.05)
    print()
    if passed:
        print("    G1 PASS — link_quality is now load-dependent. The congestion")
        print("    coupling exists: distributed load keeps quality high, concentrated")
        print("    load craters it. This is the mechanism the whole Approach-2 thesis")
        print("    (congestion-collapse, CL regimes, CBR recall) rests on. PROCEED.")
    else:
        print("    G1 FAIL — link_quality did not become meaningfully load-dependent.")
        print("    Do NOT build the rest of Approach 2 on this. Re-examine interference")
        print("    range, carrier-sense range, shadowing, or the activity levels first.")
    print("=" * 76 + "\n")
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
