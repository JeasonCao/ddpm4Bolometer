"""Turn DDPM output windows into an Octopus bin stream, so the old Octopus can run its OF on DDPM output.

Layout (all blocks exactly 10 s = 10000 samples, onset of a signal block at 1.5 s inside it):
  [zero block] [signal blocks: every DDPM output window] [residual blocks] [zero block]
  signal block  : one DDPM output window, signal trigger at block start + 1500
  residual block: DDPM output minus the mean output of its peak, noise trigger at block start + 1500.
                  It stands in for the pure-noise windows that DDPM output does not have (DDPM
                  puts a pulse into every window). Only half A (first half of each energy file)
                  is used for residual blocks; half B is the evaluation half.
Octopus then builds the ANPS from the residual blocks, the AP from the signal blocks (amplitude cut
in its config) and runs the OF on every block, with the ideal trigger file from make_trigger_file.py.

The residual is taken about the peak mean, not about a fitted amplitude x AP: a fitted amplitude
would remove exactly the component the OF measures. Selecting the events of one peak is what a
calibration line allows in real data; here the energy label stands in for that selection.

Usage: python scripts/of_compare/build_ddpm_stream.py --run 999996
Output: OFcompare_ddpm/run{run}/raw/*.bin, OFcompare_ddpm/run{run}/truth/{events,noise_windows,blocks}.csv
"""
import os, sys, csv, json, argparse
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
import numpy as np, h5py
from scripts.of_compare import generate_of_stream as GEN

D = '/mnt/d/storage/research/DDPM4bolometer'
OUT = f'{D}/OFcompare_ddpm'
SRC = {999996: (f'{D}/nonoise_out/pilot_run999996', 'waveforms_denoised_nonoise'),
       999997: (f'{D}/nonoise_out/pilot_run999997', 'waveforms_denoised_nonoise')}
L, PRE = 10000, 1500
MAX_PARTIAL = 86_400_000          # at most 24 h per bin file, split at block boundaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--energies', type=int, nargs='+', default=[15, 2615])
    a = ap.parse_args()
    src, field = SRC[a.run]
    prefix = GEN.RUNS[a.run]['prefix']
    blocks, rows = [], []                       # rows: block kind, energy, source event index, half
    for E in a.energies:
        with h5py.File(f'{src}/eval_efficiency_{E}keV.h5') as f:
            out = f[field][:].astype(np.float64)
        n = len(out); half = np.arange(n) >= n // 2
        blocks.append(out)
        rows += [('signal', E, i, 'B' if half[i] else 'A') for i in range(n)]
        res = out[~half] - out[~half].mean(0)
        blocks.append(res)
        rows += [('residual', E, i, 'A') for i in range(int((~half).sum()))]
    # signal blocks first, then residual blocks
    order = [k for k, r in enumerate(rows) if r[0] == 'signal'] + [k for k, r in enumerate(rows) if r[0] == 'residual']
    allw = np.concatenate(blocks)[order]; rows = [rows[k] for k in order]
    allw = np.concatenate([np.zeros((1, L)), allw, np.zeros((1, L))])
    nb = len(allw)
    print(f'run {a.run}: {sum(r[0] == "signal" for r in rows)} signal + {sum(r[0] == "residual" for r in rows)} residual blocks, '
          f'{nb * L / 3.6e6:.1f} h', flush=True)

    raw = f'{OUT}/run{a.run}/raw'; tru = f'{OUT}/run{a.run}/truth'
    os.makedirs(raw, exist_ok=True); os.makedirs(tru, exist_ok=True)
    per = MAX_PARTIAL // L
    files = []
    for d, s in enumerate(range(0, nb, per)):
        fn = f'{a.run:06d}_{prefix}_001_{d:03d}.bin'
        GEN.write_bin(f'{raw}/{fn}', GEN.encode(allw[s:s + per].ravel()))
        files.append(fn)
    with open(f'{tru}/events.csv', 'w', newline='') as fe, open(f'{tru}/noise_windows.csv', 'w', newline='') as fn_, \
         open(f'{tru}/blocks.csv', 'w', newline='') as fb:
        we, wn, wb = csv.writer(fe), csv.writer(fn_), csv.writer(fb)
        we.writerow(['event_id', 'energy_keV', 'onset_sample', 'half', 'source_index'])
        wn.writerow(['noise_id', 'trigger_sample', 'energy_keV', 'source_index'])
        wb.writerow(['block', 'kind', 'energy_keV', 'source_index', 'half', 'trigger_sample'])
        ne = nn = 0
        for b, (kind, E, i, h) in enumerate(rows, start=1):
            t = b * L + PRE
            wb.writerow([b, kind, E, i, h, t])
            if kind == 'signal':
                we.writerow([ne, E, t, h, i]); ne += 1
            else:
                wn.writerow([nn, t, E, i]); nn += 1
    json.dump(dict(run=a.run, source=src, field=field, prefix=prefix, partials=files, block_samples=L, pretrigger=PRE,
                   energies=a.energies, residual='DDPM output minus the mean output of its peak, half A only',
                   quantisation_uV=2 * GEN.ADC_RANGE / 2 ** 24 * 1e6),
              open(f'{tru}/run_info.json', 'w'), indent=1)
    # check: decoded stream reproduces the blocks to the quantisation step
    _, _, _, data = __import__('scripts.of_compare.verify_of_stream', fromlist=['read_bin']).read_bin(f'{raw}/{files[0]}')
    v = GEN.decode(data)[:min(per, nb) * L].reshape(-1, L)
    err = np.abs(v - allw[:len(v)]).max()
    print(f'wrote {len(files)} partials to {raw}; max decode error {err * 1e6:.2f} uV', flush=True)


if __name__ == '__main__':
    main()
