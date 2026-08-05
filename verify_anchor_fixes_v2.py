"""
verify_anchor_fixes_v2.py — execution proof for apply_anchor_fixes_v2.py.

Run from the repo root AFTER applying the patch:
    python verify_anchor_fixes_v2.py --src src

Every claim is checked by running code. The load-bearing tests are 2 and 3:
a drift anchor that only ever passes proves nothing, so the anchor logic is
exercised against a correct value, a wrong value, and the unset state; and the
second anchor is shown to catch a change the first one demonstrably misses.
"""
import argparse
import importlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='src')
ap.add_argument('--skip-slow', action='store_true',
                help='skip the four simulator runs in test 3 (~40 s)')
args = ap.parse_args()

SRC = os.path.abspath(args.src)
sys.path.insert(0, SRC)

OK, NO = "  [PASS]", "  [FAIL]"
fails = []


def check(name, cond, detail=""):
    print(f"{OK if cond else NO} {name:<60} {detail}")
    if not cond:
        fails.append(name)


print("\n" + "=" * 78)
print("  VERIFICATION — ANCHOR + PARITY FIXES v2")
print("=" * 78)

# ── 1. F4: assert instead of silent clamp ───────────────────────────────────
import link_model_v2 as L

base_ok = L.compute_link_features_v2(150.0, p_collision=0.25)
check("1a. valid p_collision accepted", base_ok is not None)

for bad, label in ((1.5, "1.5"), (-0.5, "-0.5"), (float('nan'), "NaN")):
    try:
        L.compute_link_features_v2(150.0, p_collision=bad)
        caught = False
    except AssertionError:
        caught = True
    check(f"1b. p_collision={label} raises AssertionError", caught,
          "silently clamped" if not caught else "")

# float noise at the boundary must still pass, not trip the assert
try:
    L.compute_link_features_v2(150.0, p_collision=1.0 + 1e-12)
    L.compute_link_features_v2(150.0, p_collision=-1e-12)
    noise_ok = True
except AssertionError:
    noise_ok = False
check("1c. float noise at the boundary still accepted", noise_ok,
      "tolerance 1e-9 absorbs representation error, not bugs")

# ── 2. F3: RNG parity between collision models ──────────────────────────────
import preflight_interference_check as I

rng_pos = np.random.default_rng(0)
pos = rng_pos.uniform(0, 1000, size=(30, 2))
links = I.build_feasible_links(pos, 280.0)


def run_model(model, seed=42):
    r = np.random.default_rng(seed)
    lq, per = I.link_quality_under_load(pos, 280.0, links, 0.06, r, 5.0, model)
    # the generator's post-run state is the evidence of how much it consumed
    return np.asarray(lq), np.asarray(per), r.random()


lq_s, per_s, tail_s = run_model('saturated')
lq_u, per_u, tail_u = run_model('unsaturated')
check("2a. saturated and unsaturated consume the RNG identically",
      tail_s == tail_u,
      "same generator position after the sweep -> comparison is PAIRED")
check("2b. lq now RESPONDS to the collision model (F5)",
      not np.allclose(lq_s, lq_u),
      f"mean lq {lq_s.mean():.6f} (sat) vs {lq_u.mean():.6f} (unsat)")
check("2d. ...and per responds too", not np.allclose(per_s, per_u),
      f"mean per {per_s.mean():.6f} vs {per_u.mean():.6f}")

# saturated must be byte-identical to the pre-patch implementation
PRE = os.environ.get('RECON_PRE')
if PRE and os.path.isdir(PRE):
    sys.path.insert(0, os.path.abspath(PRE))
    for m in list(sys.modules):
        if m.startswith(('preflight_interference_check', 'link_model_v2')):
            del sys.modules[m]
    import preflight_interference_check as I_pre
    r = np.random.default_rng(42)
    lq_pre, per_pre = I_pre.link_quality_under_load(pos, 280.0, links, 0.06, r,
                                                    5.0, 'saturated')
    # `per` is the F3-invariant: the unconditional draw must not perturb the
    # saturated branch's RNG consumption. lq is EXPECTED to move -- that is F5.
    check("2c. saturated `per` byte-identical to pre-patch",
          np.array_equal(np.asarray(per_pre), per_s),
          "RNG parity did not disturb the saturated branch")
    lq_pre = np.asarray(lq_pre)
    check("2e. saturated `lq` moved, and only downward (F5 fold-in)",
          (not np.array_equal(lq_pre, lq_s)) and bool((lq_s <= lq_pre + 1e-12).all()),
          f"mean {lq_pre.mean():.6f} -> {lq_s.mean():.6f}; "
          f"lq *= (1 - p_coll) can only reduce it")
    sys.path.remove(os.path.abspath(PRE))
    for m in list(sys.modules):
        if m.startswith(('preflight_interference_check', 'link_model_v2')):
            del sys.modules[m]
    sys.path.insert(0, SRC)
    import preflight_interference_check as I  # noqa: F811

# ── 3. F2: the anchor logic, and the blind spot it exists for ───────────────
import preflight_simulator_v2_check as P

check("3a. primary anchor is actor-pinned", P.REGRESSION_ACTOR == 'dijkstra',
      f"={P.REGRESSION_ACTOR!r}")
check("3b. second anchor uses a link-quality-sensitive actor",
      P.REGRESSION_LQ_ACTOR == 'spbp', f"={P.REGRESSION_LQ_ACTOR!r}")

fake = {'n_delivered': 87, 'n_generated': 280, 'network_pdr': 87 / 280}
ok, armed = P._anchor_block('REGRESSION', fake, None, None, 'dijkstra', 'f.py')
check("3c. unset  -> passes but is NOT armed", ok and not armed,
      "bootstrap must not FAIL, but must not look armed either")
ok, armed = P._anchor_block('REGRESSION', fake, 87, 280, 'dijkstra', 'f.py')
check("3d. matching anchor -> pass, armed", ok and armed)
ok, armed = P._anchor_block('REGRESSION', fake, 86, 280, 'dijkstra', 'f.py')
check("3e. WRONG anchor -> FAIL (the check can actually fail)",
      (not ok) and armed)

if not args.skip_slow:
    from simulator_v2 import FANETSimulatorV2
    base = dict(num_drones=30, area_x=1000.0, area_y=1000.0, z_min=50, z_max=150,
                comm_range=280.0, duration=40.0, speed_min=5.0, speed_max=15.0,
                pause_max=5.0, seed=42, packet_rate=1.0, interference_on=True)
    got = {}
    for actor in ('dijkstra', 'spbp'):
        for cm in ('saturated', 'unsaturated'):
            r = FANETSimulatorV2({**base, 'actor': actor,
                                  'collision_model': cm}).run()
            got[(actor, cm)] = (r['n_delivered'], r['n_generated'])
    dij_blind = got[('dijkstra', 'saturated')] == got[('dijkstra', 'unsaturated')]
    spbp_sees = got[('spbp', 'saturated')] != got[('spbp', 'unsaturated')]
    check("3f. dijkstra anchor is BLIND to the M-4 flip", dij_blind,
          f"{got[('dijkstra','saturated')][0]}/{got[('dijkstra','saturated')][1]}"
          f" -> {got[('dijkstra','unsaturated')][0]}/"
          f"{got[('dijkstra','unsaturated')][1]}")
    check("3g. spbp anchor DETECTS it", spbp_sees,
          f"{got[('spbp','saturated')][0]}/{got[('spbp','saturated')][1]}"
          f" -> {got[('spbp','unsaturated')][0]}/{got[('spbp','unsaturated')][1]}")
    check("3h. => the second anchor is load-bearing, not redundant",
          dij_blind and spbp_sees,
          "only the lq-sensitive anchor sees the change just made")
    print(f"\n    EXPECTED ANCHOR VALUES for this build:")
    print(f"        REGRESSION_DELIVERED    = {got[('dijkstra','unsaturated')][0]}")
    print(f"        REGRESSION_GENERATED    = {got[('dijkstra','unsaturated')][1]}")
    print(f"        REGRESSION_LQ_DELIVERED = {got[('spbp','unsaturated')][0]}")
    print(f"        REGRESSION_LQ_GENERATED = {got[('spbp','unsaturated')][1]}\n")

# ── 4. F1: the stale cross-reference is gone ────────────────────────────────
sim = io.open(os.path.join(SRC, 'simulator_v2.py'), encoding='utf-8').read()
check("4a. misleading REGRESSION_PDR pointer gone",
      'is recorded in preflight_simulator_v2_check.REGRESSION_PDR' not in sim,
      "the name now appears only in an explicit negation")
check("4b. ...replaced by the real constant names",
      'REGRESSION_DELIVERED' in sim and 'REGRESSION_LQ_' in sim)

print("=" * 78)
if fails:
    print(f"  {len(fails)} FAILURE(S): {fails}")
    sys.exit(1)
print("  ALL VERIFICATION CHECKS PASSED")
print("=" * 78 + "\n")
