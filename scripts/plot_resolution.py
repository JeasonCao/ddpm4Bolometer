"""
Plot energy resolution (FWHM) vs SNR for each energy point.

Reads fit-results HDF5 files produced by run_fitting.py and produces:
  - One subplot per energy point
  - Each subplot: FWHM [keV] vs SNR [dB], noisy vs denoised

Usage
-----
python scripts/plot_resolution.py \
    --fit_dir /mnt/d/storage/research/DDPM4bolometer/fit_results \
    --output  /mnt/d/storage/research/DDPM4bolometer/plots/resolution.png
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
from src.basics.plot_utils import suffix_from_model_dir, fit_energy_calibration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

N_BINS      = 20      # SNR bins for aggregation
MIN_EVENTS  = 20      # minimum events per bin to compute FWHM
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def gaussian(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def compute_fwhm_kev(amplitudes: np.ndarray, k_calib: float):
    """Fit a Gaussian to the amplitude distribution; return (FWHM, FWHM_err) in keV.

    Uses Poisson-weighted chi2 so the covariance matrix gives meaningful errors.
    Returns (nan, nan) if fit fails or too few events.
    """
    amps = amplitudes[np.isfinite(amplitudes)]
    if len(amps) < MIN_EVENTS:
        return np.nan, np.nan

    mu0    = np.median(amps)
    sigma0 = np.std(amps)
    if sigma0 == 0:
        return np.nan, np.nan

    bins   = np.linspace(mu0 - 4 * sigma0, mu0 + 4 * sigma0, 50)
    counts, edges = np.histogram(amps, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Poisson weights: floor at 1 so empty bins don't blow up chi2
    weights = np.where(counts > 0, np.sqrt(counts), 1.0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, pcov = curve_fit(
                gaussian, centers, counts,
                p0=[mu0, sigma0, counts.max()],
                sigma=weights, absolute_sigma=True,
                maxfev=5000,
            )
        sigma_V     = abs(popt[1])
        sigma_V_err = np.sqrt(abs(pcov[1, 1]))
    except Exception:
        sigma_V     = sigma0
        sigma_V_err = np.nan

    fwhm_kev     = sigma_V     * SIGMA_TO_FWHM * k_calib
    fwhm_kev_err = sigma_V_err * SIGMA_TO_FWHM * k_calib
    return float(fwhm_kev), float(fwhm_kev_err)


def process_file(h5_path: str, n_bins: int, k_n: float, k_d: float,
                 use_maxbase: bool = False, xaxis: str = 'snr'):
    """Return (x_centers, fwhm_noisy, fwhm_denoised, energy_kev).

    xaxis: 'snr'            → x = input SNR (dB)
           'baseline_fwhm'  → x = baseline FWHM = 2.355 × noise_rms × (E/A_tmpl)
    k_n / k_d: separate calibration slopes for noisy and denoised.
    """
    amp_n_key = 'amp_maxbase_noisy'    if use_maxbase else 'amp_noisy'
    amp_d_key = 'amp_maxbase_denoised' if use_maxbase else 'amp_denoised'
    with h5py.File(h5_path, 'r') as f:
        energy_kev  = float(f.attrs['energy_kev'])
        true_amp_V  = float(f.attrs['true_amp_V'])
        snr_db      = f['snr_db'][:]
        noise_rms   = f['noise_rms'][:]
        amp_noisy   = f[amp_n_key][:]
        amp_denoised= f[amp_d_key][:]
        if use_maxbase:
            conv_noisy  = np.ones(len(amp_noisy),   dtype=bool)
            conv_denois = np.ones(len(amp_denoised), dtype=bool)
        else:
            conv_noisy  = f['converged_noisy'][:]
            conv_denois = f['converged_denoised'][:]

    if xaxis == 'baseline_fwhm':
        k_tmpl = energy_kev / true_amp_V          # keV/V, template-based
        x_vals = 2.355 * noise_rms * k_tmpl       # baseline FWHM in keV
    else:
        x_vals = snr_db

    x_min = np.nanpercentile(x_vals, 1)
    x_max = np.nanpercentile(x_vals, 99)
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    centers   = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fwhm_n, fwhm_n_err = [], []
    fwhm_d, fwhm_d_err = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (x_vals >= lo) & (x_vals < hi)
        an = amp_noisy[mask & conv_noisy]
        ad = amp_denoised[mask & conv_denois]
        fn, fn_err = compute_fwhm_kev(an, k_n)
        fd, fd_err = compute_fwhm_kev(ad, k_d)
        fwhm_n.append(fn);     fwhm_n_err.append(fn_err)
        fwhm_d.append(fd);     fwhm_d_err.append(fd_err)

    return (centers,
            np.array(fwhm_n), np.array(fwhm_n_err),
            np.array(fwhm_d), np.array(fwhm_d_err),
            energy_kev)


def plot_resolution(fit_dir: str, output: str, n_bins: int, use_maxbase: bool = False,
                    xaxis: str = 'snr'):
    files = sorted(
        os.path.join(fit_dir, f)
        for f in os.listdir(fit_dir)
        if f.endswith('.h5') and 'efficiency' not in f
    )
    if not files:
        raise FileNotFoundError(f"No fit HDF5 files in {fit_dir}")

    k_n, b_n = fit_energy_calibration(files, which='noisy',    use_maxbase=use_maxbase)
    k_d, b_d = fit_energy_calibration(files, which='denoised', use_maxbase=use_maxbase)
    print(f"Calibration (noisy)   : k={k_n:.4f} keV/V, b={b_n:.4f} keV")
    print(f"Calibration (denoised): k={k_d:.4f} keV/V, b={b_d:.4f} keV")

    n_panels = len(files)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    amp_label = 'max-baseline' if use_maxbase else 'tri-exp fit'
    xaxis_label = 'Baseline FWHM (keV)' if xaxis == 'baseline_fwhm' else 'SNR (dB)'
    fig.suptitle(f'Energy Resolution vs {xaxis_label}: Before and After DDPM Denoising  [{amp_label}]',
                 fontsize=14, y=1.01)

    for ax_idx, path in enumerate(files):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]

        centers, fwhm_n, fwhm_n_err, fwhm_d, fwhm_d_err, energy = \
            process_file(path, n_bins, k_n, k_d, use_maxbase, xaxis)

        valid_n = np.isfinite(fwhm_n)
        valid_d = np.isfinite(fwhm_d)

        ax.errorbar(centers[valid_n], fwhm_n[valid_n],
                    yerr=fwhm_n_err[valid_n],
                    fmt='o-', color='steelblue', label='Before denoising',
                    linewidth=1.5, elinewidth=0.9, capsize=2)
        ax.errorbar(centers[valid_d], fwhm_d[valid_d],
                    yerr=fwhm_d_err[valid_d],
                    fmt='s-', color='tomato',    label='After denoising',
                    linewidth=1.5, elinewidth=0.9, capsize=2)

        ax.set_title(f'{energy:.0f} keV', fontsize=11)
        ax.set_xlabel(xaxis_label)
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
    parser.add_argument('--output',    default=None)
    parser.add_argument('--suffix',    default='', help='Manual suffix before .png, e.g. _run01')
    parser.add_argument('--model_cfg',   default=None, metavar='MODEL_DIR',
                        help='Model output dir; auto-builds suffix from config/history/mtime')
    parser.add_argument('--n_bins',      type=int, default=N_BINS)
    parser.add_argument('--use_maxbase', action='store_true',
                        help='Use max-baseline amplitude instead of tri-exp fitted amplitude')
    parser.add_argument('--xaxis', default='baseline_fwhm', choices=['snr', 'baseline_fwhm'],
                        help='X-axis: snr (dB) or baseline_fwhm (keV)')
    args = parser.parse_args()
    suffix = suffix_from_model_dir(args.model_cfg) if args.model_cfg else args.suffix
    if args.use_maxbase:
        suffix += '_maxbase'
    if args.xaxis == 'baseline_fwhm':
        suffix += '_blfwhm'
    output = args.output or os.path.join(_RESULTS_DIR, 'plots', 'resolution', f'resolution{suffix}.png')
    plot_resolution(args.fit_dir, output, args.n_bins, use_maxbase=args.use_maxbase, xaxis=args.xaxis)


if __name__ == '__main__':
    main()
