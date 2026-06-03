"""
Plot energy spectra before and after DDPM denoising.

For each fit-results HDF5:
  Row 1 — Overall spectrum (all converged events)
  Row 2 — Three SNR bins: low / mid / high

Usage
-----
python scripts/plot_spectra.py \
    --fit_dir /home/wsl_0vbb/DDPM4bolometer/fit_results/resolution \
    --output  /home/wsl_0vbb/DDPM4bolometer/plots/spectra.png
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


def amp_to_kev(amp_V, true_amp_V, energy_kev):
    return amp_V * (energy_kev / true_amp_V)


def plot_spectrum_panel(ax, amp_n_kev, amp_d_kev, title, energy_kev, n_bins=60):
    all_vals = np.concatenate([amp_n_kev, amp_d_kev])
    if len(all_vals) == 0:
        ax.set_title(title)
        return
    lo = np.percentile(all_vals, 1)
    hi = np.percentile(all_vals, 99)
    bins = np.linspace(lo, hi, n_bins)

    ax.hist(amp_n_kev, bins=bins, histtype='step', color='steelblue',
            lw=1.5, label=f'Before  (N={len(amp_n_kev)})')
    ax.hist(amp_d_kev, bins=bins, histtype='step', color='tomato',
            lw=1.5, label=f'After   (N={len(amp_d_kev)})')
    ax.axvline(energy_kev, color='k', ls='--', lw=1, alpha=0.6, label=f'{energy_kev:.0f} keV')
    ax.set_xlabel('Reconstructed energy (keV)', fontsize=9)
    ax.set_ylabel('Counts', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def process_file(h5_path):
    with h5py.File(h5_path, 'r') as f:
        energy_kev   = float(f.attrs['energy_kev'])
        true_amp_V   = float(f.attrs['true_amp_V'])
        snr_db       = f['snr_db'][:]
        amp_n        = f['amp_noisy'][:]
        amp_d        = f['amp_denoised'][:]
        conv_n       = f['converged_noisy'][:]
        conv_d       = f['converged_denoised'][:]
    return energy_kev, true_amp_V, snr_db, amp_n, amp_d, conv_n, conv_d


def plot_spectra(fit_dir: str, output: str):
    fit_files = sorted(
        os.path.join(fit_dir, f)
        for f in os.listdir(fit_dir)
        if f.endswith('.h5')
    )
    if not fit_files:
        raise FileNotFoundError(f"No fit HDF5 files in {fit_dir}")

    n_energies = len(fit_files)
    # 4 rows per energy: overall + 3 SNR bins
    fig, axes = plt.subplots(n_energies, 4, figsize=(18, 4.5 * n_energies))
    if n_energies == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle('Energy Spectra: Before vs After DDPM Denoising', fontsize=13)

    for row, path in enumerate(fit_files):
        energy, true_amp, snr, amp_n, amp_d, conv_n, conv_d = process_file(path)

        # Convert to keV
        akev_n = amp_to_kev(amp_n, true_amp, energy)
        akev_d = amp_to_kev(amp_d, true_amp, energy)

        # Overall
        an_all = akev_n[conv_n]
        ad_all = akev_d[conv_d]
        plot_spectrum_panel(axes[row, 0], an_all, ad_all,
                            f'{energy:.0f} keV — All SNR', energy)

        # SNR bins: low / mid / high terciles
        snr_lo, snr_hi = np.percentile(snr, [33, 67])
        snr_ranges = [
            (snr.min(),  snr_lo, 'Low SNR'),
            (snr_lo,     snr_hi, 'Mid SNR'),
            (snr_hi,     snr.max(), 'High SNR'),
        ]
        for col, (lo, hi, label) in enumerate(snr_ranges, start=1):
            mask = (snr >= lo) & (snr < hi)
            an = akev_n[mask & conv_n]
            ad = akev_d[mask & conv_d]
            title = f'{energy:.0f} keV — {label}\n({lo:.1f}–{hi:.1f} dB)'
            plot_spectrum_panel(axes[row, col], an, ad, title, energy)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit_dir', default=os.path.join(_RESULTS_DIR, 'fit_results', 'resolution'))
    parser.add_argument('--output',  default=os.path.join(_RESULTS_DIR, 'plots', 'spectra.png'))
    args = parser.parse_args()
    plot_spectra(args.fit_dir, args.output)


if __name__ == '__main__':
    main()
