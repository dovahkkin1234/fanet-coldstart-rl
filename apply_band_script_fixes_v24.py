"""apply_band_script_fixes_v24.py -- S4 display + self-describing output.

Assertion-guarded, idempotent, two anchors per file. Applies to BOTH band
scripts (they share the relevant code verbatim).

FIX 1 -- S4 display was misleading (audit Finding 4)

The A1 runs printed lines like:

    S4 no delivery collapse : 155669 -> 180891  (-16.2%, tolerance 10%)  OK
    S4 no delivery collapse : 117517 -> 193377  (-64.6%, tolerance 10%)  OK

Reading that, -64.6% next to "tolerance 10%" and "OK" looks like a badly
violated threshold that passed anyway. The LOGIC is correct: drop is computed
as (d_lo - d_hi)/d_lo, so delivery RISING with rate gives a NEGATIVE drop,
which trivially satisfies `drop <= 0.10`. A rise is not a collapse, so passing
is right. Only the presentation is wrong -- it invites exactly the
misreading a self-test exists to prevent. Now prints the direction explicitly
and only mentions the tolerance when a drop actually occurred.

FIX 2 -- output was not self-describing (audit Finding 2)

Result files embed `provenance()`, which reports config_v2.BASE (the 40 s
parity reference), NOT the operating point the run used. Four of the five A1
files therefore record no `initial_energy` at all -- the battery=8000 setting
that makes those results valid appears nowhere in the file. Adds a
`run_params` block carrying the actual duration, battery, altitude and seed
counts, alongside (not replacing) `provenance`.

fix_result_provenance_v23.py retro-fits the same block onto the files that
already exist; this patch makes every FUTURE run carry it automatically.

USAGE
    python apply_band_script_fixes_v24.py --src src --dry-run
    python apply_band_script_fixes_v24.py --src src
"""
import argparse, io, os, sys

TARGETS = ['find_usable_band.py', 'find_congestion_band_convergecast.py']
GUARD = 'v24: direction-aware'

S4_OLD = '''    print(f"  S4 no delivery collapse         : {d_lo} -> {d_hi}  "
          f"({100*drop:+.1f}%, tolerance {100*NO_COLLAPSE_TOL:.0f}%)  "
          f"{'OK' if ok else '*** COLLAPSE ***'}")'''

S4_NEW = '''    # v24: direction-aware display. The old line printed e.g.
    #   "(-64.6%, tolerance 10%)  OK"
    # which reads as a badly violated threshold that passed anyway. It was
    # actually correct -- a NEGATIVE drop means delivery ROSE with rate, which
    # is not a collapse and rightly passes -- but the presentation invited the
    # exact misreading a self-test exists to prevent.
    if drop < 0:
        print(f"  S4 no delivery collapse         : {d_lo} -> {d_hi}  "
              f"(delivery ROSE {abs(100*drop):.1f}% -- not a collapse)  "
              f"{'OK' if ok else '*** UNEXPECTED ***'}")
    else:
        print(f"  S4 no delivery collapse         : {d_lo} -> {d_hi}  "
              f"(fell {100*drop:.1f}%, tolerance {100*NO_COLLAPSE_TOL:.0f}%)  "
              f"{'OK' if ok else '*** COLLAPSE ***'}")'''


def patch_file(path, dry):
    text = io.open(path, encoding='utf-8').read()
    if GUARD in text:
        print(f'  {os.path.basename(path):<40} ALREADY APPLIED')
        return True, text

    n = text.count(S4_OLD)
    if n != 1:
        print(f'  {os.path.basename(path):<40} anchor S4 matched {n}x, expected 1  <-- SKIP')
        return False, text
    text = text.replace(S4_OLD, S4_NEW, 1)
    print(f'  {os.path.basename(path):<40} anchor S4: OK')

    # FIX 2 -- add run_params to the json.dump payload
    hits = 0
    for old, new in [
        ("json.dump({'schema': 'usable_band_v1',",
         "json.dump({'schema': 'usable_band_v1',\n"
         "                   'run_params': {'duration': args.duration,\n"
         "                                  'initial_energy': args.initial_energy,\n"
         "                                  'z_min': args.z_min, 'z_max': args.z_max,\n"
         "                                  'rates': args.rates,\n"
         "                                  'map_seeds': args.map_seeds,\n"
         "                                  'measure_seeds': args.measure_seeds,\n"
         "                                  'note': 'actual operating point of THIS run; '\n"
         "                                          'the provenance block below reports '\n"
         "                                          'config_v2.BASE, the parity reference'},"),
        ("json.dump({'schema': 'congestion_band_convergecast_v1',",
         "json.dump({'schema': 'congestion_band_convergecast_v1',\n"
         "                   'run_params': {'duration': args.duration,\n"
         "                                  'initial_energy': args.initial_energy,\n"
         "                                  'rates': args.rates,\n"
         "                                  'map_seeds': args.map_seeds,\n"
         "                                  'measure_seeds': args.measure_seeds,\n"
         "                                  'note': 'actual operating point of THIS run; '\n"
         "                                          'the provenance block below reports '\n"
         "                                          'config_v2.BASE, the parity reference'},"),
    ]:
        if text.count(old) == 1:
            text = text.replace(old, new, 1)
            hits += 1
    if hits != 1:
        print(f'  {os.path.basename(path):<40} run_params anchor matched {hits}x, expected 1  <-- SKIP')
        return False, text
    print(f'  {os.path.basename(path):<40} anchor run_params: OK')
    return True, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    results = {}
    for t in TARGETS:
        path = os.path.join(a.src, t)
        if not os.path.exists(path):
            print(f'  {t:<40} NOT FOUND  <-- SKIP')
            continue
        ok, newtext = patch_file(path, a.dry_run)
        if ok and GUARD not in io.open(path, encoding='utf-8').read():
            results[path] = newtext

    if a.dry_run:
        print(f'\n  DRY RUN OK -- {len(results)} file(s) would be written.')
        return 0
    for path, text in results.items():
        io.open(path, 'w', encoding='utf-8').write(text)
        print(f'  WROTE {path}')
    print(f'\n  {len(results)} file(s) updated.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
