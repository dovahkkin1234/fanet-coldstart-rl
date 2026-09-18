"""fix_result_provenance_v23.py -- make existing result JSONs self-describing.

THE PROBLEM (audit Finding 2)

Every band-search result file embeds `provenance()`, which reports
config_v2.BASE -- currently duration=40.0, rates=[0.5, 2.0, 4.0], z=50-150.
That is the PARITY REFERENCE, not the operating point the run actually used.
So band_dense_slow_1000s.json says "duration: 40.0" in its provenance block
while its own top-level field says the run was 1000.0s, and FOUR of the five
A1 result files record no initial_energy at all -- the 8000 battery that makes
those results valid appears nowhere in the file.

Anyone reading these files later -- including the thesis write-up -- cannot
tell from the file alone what operating point produced the numbers. That is
exactly the self-describing-provenance gap flagged in the pre-regeneration
scan, deferred to v8b, and still open.

WHAT THIS DOES

Adds a `run_params` block to each result file recording the operating point
that file was ACTUALLY produced at, taken from the command line used to
generate it. Does not touch `provenance` (which stays as the honest record of
what config_v2.BASE said at the time) and does not touch any measurement --
only adds the missing metadata.

`run_params.source` records that these values came from the invoking command,
not from the file itself, so no future reader mistakes them for something the
script measured.

USAGE
    python fix_result_provenance_v23.py --results results --dry-run
    python fix_result_provenance_v23.py --results results
"""
import argparse, json, os, sys

# Operating point each file was actually produced at, from the commands used.
KNOWN = {
    'band_dense_slow_1000s.json':  dict(duration=1000.0, initial_energy=8000.0,
                                        z_min=100, z_max=300, scenario='dense_slow',
                                        measure_seeds=15),
    'band_very_dense_1000s.json':  dict(duration=1000.0, initial_energy=8000.0,
                                        z_min=100, z_max=300, scenario='very_dense',
                                        measure_seeds=15),
    'band_medium_slow_1000s.json': dict(duration=1000.0, initial_energy=8000.0,
                                        z_min=100, z_max=300, scenario='medium_slow',
                                        measure_seeds=15),
    'band_sparse_fast_1000s.json': dict(duration=1000.0, initial_energy=8000.0,
                                        z_min=100, z_max=300, scenario='sparse_fast',
                                        measure_seeds=15),
    'band_sink50_1000s.json':      dict(duration=1000.0, initial_energy=8000.0,
                                        scenario='sink_50', measure_seeds=15),
    'energy_range.json':           dict(duration=1000.0, z_min=100, z_max=300,
                                        scenario='dense_slow', note='battery sweep'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    if not os.path.isdir(a.results):
        print(f'  ERROR: {a.results} not found'); return 1

    touched, skipped = 0, 0
    for fname, params in KNOWN.items():
        path = os.path.join(a.results, fname)
        if not os.path.exists(path):
            print(f'  {fname:<32} NOT FOUND -- skipped')
            skipped += 1
            continue
        with open(path) as f:
            d = json.load(f)
        if 'run_params' in d:
            print(f'  {fname:<32} already has run_params -- skipped')
            skipped += 1
            continue
        d['run_params'] = {
            **params,
            'source': 'recorded retroactively from the generating command line '
                      '(v23); provenance block above reflects config_v2.BASE at '
                      'run time, which is the parity reference, NOT this run',
        }
        print(f'  {fname:<32} + run_params  duration={params.get("duration")} '
              f'energy={params.get("initial_energy", "n/a")}')
        if not a.dry_run:
            with open(path, 'w') as f:
                json.dump(d, f, indent=2)
        touched += 1

    print(f'\n  {touched} file(s) updated, {skipped} skipped')
    if a.dry_run:
        print('  DRY RUN -- nothing written.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
