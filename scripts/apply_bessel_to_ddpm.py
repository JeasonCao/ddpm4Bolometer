"""Low-pass the DDPM zero-init outputs (or, with --source noisy, the raw noisy windows) with a Bessel
filter and store them in a separate tree.

Filter: exactly the simulated readout chain, reused from src/basics/filters.apply_bessel_decimate:
6th-order Bessel, "120 Hz" with scipy norm='phase' (real -3 dB point 69.35 Hz), designed and applied
causally at 10 kHz, then every 10th sample kept. The 1 kHz DDPM output is first brought to 10 kHz
with scipy.signal.resample_poly(x, 10, 1); the whole chain then matches the simulator's filter to
<= 0.1 dB up to 300 Hz (checked with white noise). Zero initial filter state; the DDPM pre-trigger
is ~0 so there is no start-up transient.

Output files contain only the filtered field; noisy waveforms, clean template, noise params
etc. are HDF5 external links to the source file, so run_fitting.py can read them unchanged
and nothing is duplicated. The source files are opened read-only and never modified.

Usage:
  python scripts/apply_bessel_to_ddpm.py --group efficiency_7p5mV                 # DDPM + Bessel
  python scripts/apply_bessel_to_ddpm.py --group efficiency_7p5mV --source noisy  # Bessel only, no DDPM
Output:
  /mnt/d/storage/research/DDPM4bolometer/eval_data_bessel/{tag}/{group}/eval_efficiency_*keV.h5
  with dataset waveforms_denoised_zeroinit_{tag}, tag = bessel_sim (simulator chain, 120 Hz phase-norm, order 6)
  --source noisy: eval_data_bessel/bessel_sim_noisy/{group}/..., dataset waveforms_noisy_bessel_sim
"""
import os, sys, glob, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, h5py
from scipy.signal import resample_poly
from src.basics.filters import apply_bessel_decimate

D = '/mnt/d/storage/research/DDPM4bolometer'
SRC = f'{D}/eval_data_holdout_denoised'
FIELD = 'waveforms_denoised_zeroinit'
LINKED = ['waveforms_noisy', 'clean_template', 'noise_params', 'noise_rms', 'snr_db']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True)
    ap.add_argument('--chunk', type=int, default=250)
    ap.add_argument('--source', choices=['ddpm', 'noisy'], default='ddpm')
    a = ap.parse_args()
    tag = 'bessel_sim'
    field = FIELD if a.source == 'ddpm' else 'waveforms_noisy'
    out_dir = f'{D}/eval_data_bessel/{tag}' + ('_noisy' if a.source == 'noisy' else '') + f'/{a.group}'
    os.makedirs(out_dir, exist_ok=True)

    def filt(x):
        up = resample_poly(x, 10, 1, axis=1)
        return np.stack([apply_bessel_decimate(r, 10000.0, 1000.0, 120.0, 6) for r in up])[:, :x.shape[1]]
    out_field = f'{field}_{tag}'
    for src in sorted(glob.glob(f'{SRC}/{a.group}/eval_efficiency_*keV.h5')):
        dst = f'{out_dir}/{os.path.basename(src)}'
        t0 = time.time()
        with h5py.File(src, 'r') as fi, h5py.File(dst + '.tmp', 'w') as fo:
            for k, v in fi.attrs.items():
                fo.attrs[k] = v
            fo.attrs.update(dict(bessel_fc_Hz=120.0, bessel_order=6, bessel_norm='phase', bessel_minus3dB_Hz=69.35,
                                 bessel_chain='resample_poly x10 -> apply_bessel_decimate(10 kHz -> 1 kHz)',
                                 bessel_source_field=field, bessel_source_file=src))
            for k in LINKED:
                fo[k] = h5py.ExternalLink(src, k)
            w = fi[field]; n = len(w)
            ds = fo.create_dataset(out_field, shape=w.shape, dtype=np.float32, compression='gzip',
                                   chunks=(min(64, n), w.shape[1]))
            for s in range(0, n, a.chunk):
                ds[s:s + a.chunk] = filt(w[s:s + a.chunk].astype(np.float64)).astype(np.float32)
        os.replace(dst + '.tmp', dst)
        print(f'{os.path.basename(src)}: {n} windows -> {dst} ({time.time() - t0:.0f} s)', flush=True)
    print('BESSEL_DONE', out_dir)


if __name__ == '__main__':
    main()
