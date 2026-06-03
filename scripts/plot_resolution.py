"""
Plot energy resolution (FWHM) vs SNR for each energy point.

Reads fit-results HDF5 files produced by run_fitting.py and produces:
  - One subplot per energy point
  - Each subplot: FWHM [keV] vs SNR [dB], noisy vs denoised

Usage
-----
python scripts/plot_resolution.py \
    --fit_dir /home/wsl_0vbb/DDPM4bolometer/fit_results \
    --output  /home/wsl_0vbb/DDPM4bolometer/plots/resolution.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import warnings
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

N_BINS      = 20      # SNR bins for aggregation
MIN_EVENTS  = 20      # minimum events per bin to compute FWHM
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def gaussian(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def compute_fwhm_kev(amplitudes: np.ndarray, true_amp_V: float,
                     energy_kev: float) -> float:
    """Fit a Gaussian to the amplitude distribution; return FWHM in keV.

    Converts V → keV assuming a linear amplitude–energy relationship.
    """
    amps = amplitudes[np.isfinite(amplitudes)]
    if len(amps) < MIN_EVENTS:
        return np.nan

    mu0    = np.median(amps)
    sigma0 = np.std(amps)
    if sigma0 == 0:
        return np.nan

    bins   = np.linspace(mu0 - 4 * sigma0, mu0 + 4 * sigma0, 50)
    counts, edges = np.histogram(amps, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(
                gaussian, centers, counts,
                p0=[mu0, sigma0, counts.max()],
                maxfev=5000,
            )
        sigma_V = abs(popt[1])
    except Exception:
        sigma_V = sigma0

    # Convert: FWHM_keV = FWHM_V * (energy_kev / true_amp_V)
    if true_amp_V <= 0:
        return np.nan
    fwhm_V   = sigma_V * SIGMA_TO_FWHM
    fwhm_kev = fwhm_V * (energy_kev / true_amp_V)
    return float(fwhm_kev)


def process_file(h5_path: str, n_bins: int):
    """Return (snr_centers, fwhm_noisy, fwhm_denoised, energy_kev)."""
    with h5py.File(h5_path, 'r') as f:
        energy_kev  = float(f.attrs['energy_kev'])
        true_amp_V  = float(f.attrs['true_amp_V'])
        snr_db      = f['snr_db'][:]
        amp_noisy   = f['amp_noisy'][:]
        amp_denoised= f['amp_denoised'][:]
        conv_noisy  = f['converged_noisy'][:]
        conv_denois = f['converged_denoised'][:]

    snr_min = np.nanpercentile(snr_db, 1)
    snr_max = np.nanpercentile(snr_db, 99)
    bin_edges = np.linspace(snr_min, snr_max, n_bins + 1)
    centers   = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fwhm_n, fwhm_d = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (snr_db >= lo) & (snr_db < hi)
        an = amp_noisy[mask & conv_noisy]
        ad = amp_denoised[mask & conv_denois]
        fwhm_n.append(compute_fwhm_kev(an, true_amp_V, energy_kev))
        fwhm_d.append(compute_fwhm_kev(ad, true_amp_V, energy_kev))

    return centers, np.array(fwhm_n), np.array(fwhm_d), energy_kev


def plot_resolution(fit_dir: str, output: str, n_bins: int):
    files = sorted(
        os.path.join(fit_dir, f)
        for f in os.listdir(fit_dir)
        if f.endswith('.h5') and 'efficiency' not in f
    )
    if not files:
        raise FileNotFoundError(f"No fit HDF5 files in {fit_dir}")

    n_panels = len(files)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle('Energy Resolution vs SNR: Before and After DDPM Denoising',
                 fontsize=14, y=1.01)

    for ax_idx, path in enumerate(files):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]

        centers, fwhm_n, fwhm_d, energy = process_file(path, n_bins)

        valid_n = np.isfinite(fwhm_n)
        valid_d = np.isfinite(fwhm_d)

        ax.plot(centers[valid_n], fwhm_n[valid_n],
                'o-', color='steelblue', label='Before denoising', linewidth=1.5)
        ax.plot(centers[valid_d], fwhm_d[valid_d],
                's-', color='tomato',    label='After denoising',  linewidth=1.5)

        ax.set_title(f'{energy:.0f} keV', fontsize=11)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('FWHM (keV)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # Hide unused panels
    for ax_idx in range(len(files), nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved → {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot energy resolution vs SNR")
    parser.add_argument('--fit_dir', default=os.path.join(_RESULTS_DIR, 'fit_results', 'resolution'))
    parser.add_argument('--output',  default=os.path.join(_RESULTS_DIR, 'plots', 'resolution.png'))
    parser.add_argument('--n_bins',  type=int, default=N_BINS)
    args = parser.parse_args()
    plot_resolution(args.fit_dir, args.output, args.n_bins)


if __name__ == '__main__':
    main()
