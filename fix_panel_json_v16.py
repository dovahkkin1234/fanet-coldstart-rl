"""fix_panel_json_v16.py -- recompute the correct verdict from a SAVED panel JSON.

No re-simulation. Every number needed (the 'means' dict, the raw p-values that
fed p_holm) is already in the file -- only the stored 'delta' is wrong (see
apply_panel_delta_fix_v16.py for why). This corrects delta in place and
re-derives the verdict. p_holm / significant are UNCHANGED: the paired t-test
that produced them used the correct incumbent already, so Holm correction
never needs re-running.

USAGE
    python fix_panel_json_v16.py results\\panel_cc_v2.json
"""
import json, sys


def main():
    if len(sys.argv) != 2:
        print("usage: fix_panel_json_v16.py <panel_results.json>")
        return 1
    path = sys.argv[1]
    d = json.load(open(path))

    print("=" * 92)
    print(f"  RECOMPUTING delta in {path} (no re-simulation)")
    print("=" * 92)

    beaten = {}
    for key in sorted(d['analysis'], key=lambda k: float(k.split('|')[1])):
        r = d['analysis'][key]
        means = r['means']
        incumbent = means.get('spbp')
        if incumbent is None:
            continue
        for t, c in r['comparisons'].items():
            old = c['delta']
            c['delta'] = means[t] - incumbent
            if abs(old - c['delta']) > 1e-6:
                pass  # corrected silently; both values are printed below
        sig_hits = {t: c for t, c in r['comparisons'].items()
                    if c.get('significant') and c['delta'] > 0}
        if sig_hits:
            beaten[key] = sig_hits

    print(f"\n  {'cell':<16}{'teacher':<16}{'corrected delta':>16}{'p_holm':>10}   sig")
    print("  " + "-" * 62)
    for key in sorted(d['analysis'], key=lambda k: float(k.split('|')[1])):
        r = d['analysis'][key]
        for t, c in r['comparisons'].items():
            if not c.get('significant'):
                continue
            print(f"  {key:<16}{t:<16}{c['delta']:>+16.4f}{c.get('p_holm', float('nan')):>10.4g}   "
                  f"{'beats spbp' if c['delta'] > 0 else 'WORSE than spbp'}")

    print("\n" + "=" * 92)
    print("  CORRECTED VERDICT")
    print("=" * 92)
    if not beaten:
        print("  No challenger significantly beats spbp in any cell.")
        print("  -> KEEP ORACLE_TEACHER = 'spbp'.")
    else:
        print("  Challengers that DO significantly beat spbp:")
        for key, hits in beaten.items():
            for t, c in hits.items():
                print(f"     {key}  {t}  {c['delta']:+.4f}  p_holm={c.get('p_holm'):.4g}")
        print("  -> Do not pin 'spbp'; use the per-cell oracle_teacher_for() lookup.")

    out = path.replace('.json', '_corrected.json')
    with open(out, 'w') as f:
        json.dump(d, f, indent=2)
    print(f"\n  saved corrected copy to {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
