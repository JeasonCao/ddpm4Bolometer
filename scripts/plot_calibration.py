"""
Plot energy calibration: mean amplitude vs true energy for all resolution points.

Two panels — tri-exp fit amplitude and max-baseline amplitude — each showing
noisy and denoised means with the zero-intercept global fit line overlaid.

Usage
-----
python scripts/plot_calibration.py \
    --fit_dir  results/fit_results/resolution \
    --model_cfg /path/to/model_output_cloud
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

from src.basics.plot_utils import suffix_from_model_dir, fit_energy_calibration


def load_calibration_points(fit_files, use_maxbase=False):
    """Return arrays (energy_kev, mean_amp_noisy, mean_amp_denoised) for all files."""
    amp_n_key  = 'amp_maxbase_noisy'    if use_maxbase else 'amp_noisy'
    amp_d_key  = 'amp_maxbase_denoised' if use_maxbase else 'amp_denoised'
    conv_n_key = None if use_maxbase else 'converged_noisy'
    conv_d_key = None if use_maxbase else 'converged_denoised'

    energies, means_n, means_d = [], [], []
    for path in fit_files:
        with h5py.File(path, 'r') as f:
            e   = float(f.attrs['energy_kev'])
            an  = f[amp_n_key][:]
            ad  = f[amp_d_key][:]
            mn  = np.ones(len(an), dtype=bool) if use_maxbase else f[conv_n_key][:].astype(bool)
            md  = np.ones(len(ad), dtype=bool) if use_maxbase else f[conv_d_key][:].astype(bool)
        energies.append(e)
        means_n.append(np.nanmean(an[mn]))
        means_d.append(np.nanmean(ad[md]))
    idx = np.argsort(energies)
    return (np.array(energies)[idx],
            np.array(means_n)[idx],
            np.array(means_d)[idx])


def plot_calib_panel(ax, energies, means_n, means_d,
                     k_n, b_n, k_d, b_d, title, amp_unit='V'):
    amp_min = min(means_n.min(), means_d.min())
    amp_max = max(means_n.max(), means_d.max())
    amp_line = np.linspace(max(0, amp_min * 0.95), amp_max * 1.05, 200)

    ax.plot(amp_line, k_n * amp_line + b_n, color='steelblue', ls='--', lw=1.2,
            label=f'Fit (before)  k={k_n:.1f}, b={b_n:.1f} keV')
    ax.plot(amp_line, k_d * amp_line + b_d, color='tomato', ls='--', lw=1.2,
            label=f'Fit (after)   k={k_d:.1f}, b={b_d:.1f} keV')
    ax.scatter(means_n, energies, color='steelblue', s=50, zorder=5,
               label='Before denoising')
    ax.scatter(means_d, energies, color='tomato', marker='s', s=50, zorder=5,
               label='After denoising')

    # residual of each point relative to its own calibration line
    for e, an, ad in zip(energies, means_n, means_d):
        res_n = (k_n * an + b_n - e) / e * 100
        res_d = (k_d * ad + b_d - e) / e * 100
        ax.annotate(f'{res_n:+.1f}%', xy=(an, e), xytext=(4, 0),
                    textcoords='offset points', fontsize=6, color='steelblue', va='center')
        ax.annotate(f'{res_d:+.1f}%', xy=(ad, e), xytext=(4, -8),
                    textcoords='offset points', fontsize=6, color='tomato', va='center')

    ax.set_xlabel(f'Mean amplitude ({amp_unit})', fontsize=10)
    ax.set_ylabel('True energy (keV)', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_calibration(fit_dir, output, model_cfg=None, suffix=''):
    fit_files = sorted(
        os.path.join(fit_dir, f)
        for f in os.listdir(fit_dir)
        if f.endswith('.h5') and 'efficiency' not in f
    )
    if not fit_files:
        raise FileNotFoundError(f"No resolution fit HDF5 files in {fit_dir}")

    k_fit_n, b_fit_n = fit_energy_calibration(fit_files, which='noisy',    use_maxbase=False)
    k_fit_d, b_fit_d = fit_energy_calibration(fit_files, which='denoised', use_maxbase=False)
    k_max_n, b_max_n = fit_energy_calibration(fit_files, which='noisy',    use_maxbase=True)
    k_max_d, b_max_d = fit_energy_calibration(fit_files, which='denoised', use_maxbase=True)

    print(f"Tri-exp  noisy   : k={k_fit_n:.4f}, b={b_fit_n:.4f}")
    print(f"Tri-exp  denoised: k={k_fit_d:.4f}, b={b_fit_d:.4f}")
    print(f"Maxbase  noisy   : k={k_max_n:.4f}, b={b_max_n:.4f}")
    print(f"Maxbase  denoised: k={k_max_d:.4f}, b={b_max_d:.4f}")

    e_fit, mn_fit, md_fit = load_calibration_points(fit_files, use_maxbase=False)
    e_max, mn_max, md_max = load_calibration_points(fit_files, use_maxbase=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Energy Calibration: Mean Amplitude vs True Energy\n'
                 '(separate linear fits for before/after denoising)', fontsize=13)

    plot_calib_panel(axes[0], e_fit, mn_fit, md_fit,
                     k_fit_n, b_fit_n, k_fit_d, b_fit_d, 'Tri-exp fit amplitude')
    plot_calib_panel(axes[1], e_max, mn_max, md_max,
                     k_max_n, b_max_n, k_max_d, b_max_d, 'Max-baseline amplitude')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {output}")


def main():
    parser = argparse.ArgumentParser(description='Plot energy calibration')
    parser.add_argument('--fit_dir',   default=os.path.join(_RESULTS_DIR, 'fit_results', 'resolution'))
    parser.add_argument('--output',    default=None)
    parser.add_argument('--suffix',    default='')
    parser.add_argument('--model_cfg', default=None, metavar='MODEL_DIR')
    args = parser.parse_args()

    suffix = suffix_from_model_dir(args.model_cfg) if args.model_cfg else args.suffix
    output = args.output or os.path.join(
        _RESULTS_DIR, 'plots', 'resolution', f'calibration{suffix}.png')
    plot_calibration(args.fit_dir, output)


if __name__ == '__main__':
    main()
