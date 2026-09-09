"""
Plot a single-energy reconstructed-energy spectrum (before vs after denoising)
with the Gaussian fit overlaid, as a worked methodology example for the paper
(Sec. 4.3, "single energy spectrum + fit" figure).

Reads one fit-results HDF5 (must already have gauss_* attrs from
run_gauss_fit.py) and produces one clean single-panel figure.

Calibration: uses the trivial single-point conversion k = energy_kev /
true_amp_V (amplitude-proportional-to-energy assumption), consistent with
the "baseline_fwhm" calculation in plot_resolution.py. This is intentional
for a single-energy illustration and is NOT the multi-point global
calibration used elsewhere in the efficiency/resolution pipelines.

Usage
-----
python scripts/plot_single_energy_example.py \
    --fit_file results/fit_results/efficiency_7p5mV/fit_efficiency_15keV.h5 \
    --output   results/plots/reconstruction_quality/spectrum_example_15keV_7p5mV.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def gaussian(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit_file', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--n_bins', type=int, default=60)
    ap.add_argument('--use_maxbase', action='store_true')
    args = ap.parse_args()

    tag = 'maxbase' if args.use_maxbase else 'triexp'
    amp_n_key = 'amp_maxbase_noisy'    if args.use_maxbase else 'amp_noisy'
    amp_d_key = 'amp_maxbase_denoised' if args.use_maxbase else 'amp_denoised'

    with h5py.File(args.fit_file, 'r') as f:
        energy_kev = float(f.attrs['energy_kev'])
        true_amp_V = float(f.attrs['true_amp_V'])
        amp_noisy    = f[amp_n_key][:]
        amp_denoised = f[amp_d_key][:]
        if not args.use_maxbase:
            conv_n = f['converged_noisy'][:]
            conv_d = f['converged_denoised'][:]
            amp_noisy    = amp_noisy[conv_n]
            amp_denoised = amp_denoised[conv_d]

        mu_n    = float(f.attrs[f'gauss_mu_{tag}_noisy'])
        sigma_n = float(f.attrs[f'gauss_sigma_{tag}_noisy'])
        mu_d    = float(f.attrs[f'gauss_mu_{tag}_denoised'])
        sigma_d = float(f.attrs[f'gauss_sigma_{tag}_denoised'])
        n_n = int(f.attrs[f'gauss_n_{tag}_noisy'])
        n_d = int(f.attrs[f'gauss_n_{tag}_denoised'])

    k = energy_kev / true_amp_V   # keV/V, single-point (template) calibration

    e_noisy    = amp_noisy    * k
    e_denoised = amp_denoised * k
    mu_n_kev, sigma_n_kev = mu_n * k, sigma_n * k
    mu_d_kev, sigma_d_kev = mu_d * k, sigma_d * k
    fwhm_n = sigma_n_kev * SIGMA_TO_FWHM
    fwhm_d = sigma_d_kev * SIGMA_TO_FWHM

    lo = min(e_noisy.min(), e_denoised.min())
    hi = max(e_noisy.max(), e_denoised.max())
    pad = 0.05 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, args.n_bins)
    bin_width = bins[1] - bins[0]

    fig, ax = plt.subplots(figsize=(7, 5.2))

    ax.hist(e_noisy, bins=bins, histtype='step', color='steelblue', lw=1.6,
            label=f'Before denoising (N={n_n})')
    ax.hist(e_denoised, bins=bins, histtype='step', color='tomato', lw=1.6,
            label=f'After denoising (N={n_d})')

    x_dense = np.linspace(bins[0], bins[-1], 500)
    A_n = n_n * bin_width / (sigma_n_kev * np.sqrt(2 * np.pi))
    A_d = n_d * bin_width / (sigma_d_kev * np.sqrt(2 * np.pi))
    ax.plot(x_dense, gaussian(x_dense, mu_n_kev, sigma_n_kev, A_n),
            color='navy', lw=1.4, ls='--',
            label=f'Gaussian fit (before): FWHM = {fwhm_n:.2f} keV')
    ax.plot(x_dense, gaussian(x_dense, mu_d_kev, sigma_d_kev, A_d),
            color='darkred', lw=1.4, ls='--',
            label=f'Gaussian fit (after): FWHM = {fwhm_d:.3f} keV')

    ax.axvline(energy_kev, color='gray', ls=':', lw=1.2, label=f'True energy = {energy_kev:.0f} keV')

    ax.set_xlabel('Reconstructed energy (keV)')
    ax.set_ylabel('Counts')
    ax.set_title(f'Energy spectrum before/after denoising — {energy_kev:.0f} keV example\n'
                 f'({tag} fit)')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f'Saved -> {args.output}')
    print(f'  FWHM before = {fwhm_n:.3f} keV, after = {fwhm_d:.3f} keV '
          f'(improvement {fwhm_n/fwhm_d:.1f}x)')


if __name__ == '__main__':
    main()
