"""
Plot small-signal reconstruction efficiency: before vs after DDPM denoising.

Reads a fit-results HDF5 from the efficiency dataset and produces two panels:
  Panel 1 — Fit convergence rate (%) before and after denoising
  Panel 2 — Amplitude deviation distribution before and after denoising

Usage
-----
python scripts/plot_efficiency.py \
    --fit_file /home/wsl_0vbb/DDPM4bolometer/fit_results/fit_efficiency_100keV.h5 \
    --output   /home/wsl_0vbb/DDPM4bolometer/plots/efficiency_100keV.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_efficiency(fit_file: str, output: str):
    with h5py.File(fit_file, 'r') as f:
        energy_kev     = float(f.attrs['energy_kev'])
        true_amp_V     = float(f.attrs['true_amp_V'])
        amp_dev_max    = float(f.attrs['amp_dev_max'])
        snr_db         = f['snr_db'][:]

        conv_n  = f['converged_noisy'][:]
        conv_d  = f['converged_denoised'][:]
        eff_n   = f['efficient_noisy'][:]
        eff_d   = f['efficient_denoised'][:]
        dev_n   = f['amp_dev_noisy'][:]
        dev_d   = f['amp_dev_denoised'][:]

    n_total      = len(snr_db)
    snr_mean     = float(np.nanmean(snr_db))
    conv_rate_n  = conv_n.mean() * 100
    conv_rate_d  = conv_d.mean() * 100
    eff_rate_n   = eff_n.mean()  * 100
    eff_rate_d   = eff_d.mean()  * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f'Small-Signal Reconstruction: {energy_kev:.0f} keV  '
                 f'(SNR ≈ {snr_mean:.1f} dB,  N={n_total})',
                 fontsize=13)

    # ── Panel 1: bar chart of convergence and efficiency rates ───────────────
    ax = axes[0]
    labels   = ['Convergence\nrate', f'Efficiency\n(|ΔA|/A < {amp_dev_max:.0%})']
    vals_n   = [conv_rate_n, eff_rate_n]
    vals_d   = [conv_rate_d, eff_rate_d]
    x        = np.arange(len(labels))
    width    = 0.35

    bars_n = ax.bar(x - width / 2, vals_n, width, label='Before denoising',
                    color='steelblue', alpha=0.85)
    bars_d = ax.bar(x + width / 2, vals_d, width, label='After denoising',
                    color='tomato', alpha=0.85)

    for bar in list(bars_n) + list(bars_d):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Rate (%)')
    ax.set_title('Reconstruction rates')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 2: amplitude-deviation histogram ───────────────────────────────
    ax = axes[1]
    dev_max_plot = 1.0    # show up to 100% deviation
    bins = np.linspace(0, dev_max_plot, 50)

    dev_n_clipped = np.clip(dev_n[conv_n],  0, dev_max_plot)
    dev_d_clipped = np.clip(dev_d[conv_d],  0, dev_max_plot)

    ax.hist(dev_n_clipped, bins=bins, density=True, alpha=0.6,
            color='steelblue', label=f'Before  (converged {conv_rate_n:.0f}%)')
    ax.hist(dev_d_clipped, bins=bins, density=True, alpha=0.6,
            color='tomato',    label=f'After   (converged {conv_rate_d:.0f}%)')
    ax.axvline(amp_dev_max, color='black', linestyle='--', linewidth=1,
               label=f'Efficiency cut ({amp_dev_max:.0%})')

    ax.set_xlabel('|Reconstructed − True| / True')
    ax.set_ylabel('Density')
    ax.set_title('Amplitude deviation (converged events only)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved → {output}")
    print(f"Summary  |  convergence: {conv_rate_n:.1f}% → {conv_rate_d:.1f}%  "
          f"|  efficiency: {eff_rate_n:.1f}% → {eff_rate_d:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Plot small-signal reconstruction efficiency")
    parser.add_argument('--fit_file', required=True)
    parser.add_argument('--output',   default=None,
                        help='Output path; defaults to results/plots/efficiency_<stem>.png')
    args = parser.parse_args()
    output = args.output
    if output is None:
        stem = os.path.splitext(os.path.basename(args.fit_file))[0]
        output = os.path.join(_RESULTS_DIR, 'plots', f'efficiency_{stem[4:]}.png')
    plot_efficiency(args.fit_file, output)


if __name__ == '__main__':
    main()
