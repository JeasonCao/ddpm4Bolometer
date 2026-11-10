"""
2026-09-22: paper figure -- reconstructed energy spectrum at a single low
energy (15 keV by default), before and after denoising, one panel per noise
level.

Differs from plot_spectra.py, which walks a whole fit_dir and emits a
multi-panel/SNR-binned overview of the `resolution` eval set. This one is the
single figure the paper text refers to: one energy, both noise levels side by
side, calibrated with the same 5-point zero-intercept convention as every
other figure.

Amplitudes are calibrated with the shared zero-intercept global calibration
(src/basics/plot_utils.fit_global_calibration_zero_intercept) rather than the
per-file `true_amp_V` ratio, so the x axis matches the resolution and
efficiency figures exactly.

Usage
-----
python -u scripts/plot_spectrum_15keV_zeroinit.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.basics.plot_utils import fit_global_calibration_zero_intercept
from scripts.run_gauss_fit import fit_gaussian

# Same convention as plot_reconstruction_quality.py / plot_combined_efficiency.py:
# baseline resolution FWHM = 2.355 * noise_rms_V * k.
_NOISE_RMS_V = {'1p382mV': 0.001382, '7p5mV': 0.0075}
_NOISE_LABEL = {'1p382mV': '1.382 mV', '7p5mV': '7.5 mV'}

DEFAULT_GROUPS = [
    ('1p382mV', os.path.join(_RESULTS_DIR, 'fit_results', '1p382mV_zeroinit')),
    ('7p5mV',   os.path.join(_RESULTS_DIR, 'fit_results', '7p5mV_zeroinit')),
]


def panel(ax, fit_dir, noise_tag, energy_kev, n_bins):
    path = os.path.join(fit_dir, f'fit_efficiency_{energy_kev:.0f}keV.h5')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    k, _ = fit_global_calibration_zero_intercept(fit_dir, 'triexp')
    with h5py.File(path, 'r') as f:
        amp_n = f['amp_noisy'][:].astype(float) * k
        amp_d = f['amp_denoised'][:].astype(float) * k

    # Plot the full samples -- no chi2 gating -- so the histogram shows every
    # event that went into the resolution number.
    lo = np.percentile(np.concatenate([amp_n, amp_d]), 0.5)
    hi = np.percentile(np.concatenate([amp_n, amp_d]), 99.5)
    bins = np.linspace(lo, hi, n_bins)

    gn, gd = fit_gaussian(amp_n), fit_gaussian(amp_d)
    fwhm_n, fwhm_d = 2.355 * gn['sigma'], 2.355 * gd['sigma']

    ax.hist(amp_n, bins=bins, histtype='step', color='#2f6f9f', lw=1.6,
            label=f'Before   FWHM = {fwhm_n:.2f} keV')
    ax.hist(amp_d, bins=bins, histtype='step', color='#c2603c', lw=1.6,
            label=f'After    FWHM = {fwhm_d:.2f} keV')
    ax.axvline(energy_kev, color='k', ls='--', lw=1.0, alpha=0.7,
               label=f'True energy = {energy_kev:.0f} keV')

    # Canonical slope, not this figure's k — see plot_utils, so every figure
    # in the paper prints the same baseline resolution for a given noise level.
    from src.basics.plot_utils import baseline_resolution_kev
    base = baseline_resolution_kev(_NOISE_RMS_V[noise_tag])
    ax.set_title(f'{_NOISE_LABEL[noise_tag]} RMS baseline noise\n'
                 f'(baseline resolution = {base:.2f} keV)', fontsize=10)
    ax.set_xlabel('Reconstructed energy (keV)', fontsize=10)
    ax.set_ylabel('Counts', fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    return fwhm_n, fwhm_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--energy', type=float, default=15.0)
    ap.add_argument('--n_bins', type=int, default=70)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    out = args.output or os.path.join(
        _RESULTS_DIR, 'plots', 'paper_zeroinit',
        f'spectrum_{args.energy:.0f}keV_zeroinit.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (tag, fit_dir), lab in zip(axes, DEFAULT_GROUPS, 'ab'):
        fn, fd = panel(ax, fit_dir, tag, args.energy, args.n_bins)
        ax.text(0.015, 0.97, f'({lab})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
        print(f'{tag:8s}  {args.energy:.0f} keV   before {fn:.3f} keV -> after {fd:.3f} keV'
              f'   ({fn/fd:.1f}x better)')

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
