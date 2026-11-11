"""
Illustrative figures for the ΔE/E shape diagnostics (skewness/kurtosis/outlier-tail
analysis, see results/analysis/reco_efficiency_shape_diagnostics_summary.md).

Shows the "core narrows but a stubborn tail survives" effect at low energy that a
single Gaussian resolution number hides but a fixed-threshold pass-rate does not.

3 figures:
  1. Low-energy ΔE/E histograms (1 keV, 5 keV @ 1.2mV) — noisy vs denoised, log-y
  2. High-energy ΔE/E histograms (239 keV, 511 keV @ 1.2mV) — clean contrast case
  3. 4-panel trend plot: |skew|, |excess kurtosis|, outlier_frac_20%, chi2/ndf vs energy

No pandas dependency (not installed in the project env) — CSV loaded as plain dicts.

Usage
-----
python scripts/plot_shape_diagnostic_illustration.py \
    --fit_dir_1p2mV <snapshot>/efficiency_1p2mV \
    --fit_dir_7p5mV <snapshot>/efficiency_7p5mV \
    --csv <snapshot>/reco_efficiency_shape_diagnostics.csv \
    --output_dir results/analysis
"""

import argparse
import csv as csvmod
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLOR_NOISY = 'steelblue'
COLOR_DENOISED = 'tomato'

_NUMERIC_COLS = ('energy_kev', 'N_success', 'skew', 'skew_se', 'kurt', 'kurt_se',
                  'outlier_frac_20', 'outlier_frac_30', 'chi2ndf')


def _load_csv(path):
    rows = []
    with open(path, newline='') as f:
        for r in csvmod.DictReader(f):
            for k in _NUMERIC_COLS:
                r[k] = float(r[k])
            rows.append(r)
    return rows


def _row(rows, energy_kev, noise, curve):
    for r in rows:
        if r['energy_kev'] == energy_kev and r['noise'] == noise and r['curve'] == curve:
            return r
    raise KeyError(f'no row for energy={energy_kev} noise={noise} curve={curve}')


def _load_delta_e(path):
    with h5py.File(path, 'r') as f:
        true_amp = float(f.attrs['true_amp_V'])
        amp_n = f['amp_noisy'][:].astype(float)
        amp_d = f['amp_denoised'][:].astype(float)
        succ_n = f['success_noisy'][:].astype(bool)
        succ_d = f['success_denoised'][:].astype(bool)
        e = float(f.attrs['energy_kev'])
    de_n = amp_n[succ_n] / true_amp - 1.0
    de_d = amp_d[succ_d] / true_amp - 1.0
    return e, de_n, de_d


def _hist_panel(ax, de_n, de_d, e_kev, noise_str, row_n=None, row_d=None):
    bins = np.linspace(-1.0, 1.5, 120)
    ax.hist(de_n, bins=bins, histtype='step', color=COLOR_NOISY, lw=1.4,
            label='Before denoising', density=True)
    ax.hist(de_d, bins=bins, histtype='step', color=COLOR_DENOISED, lw=1.4,
            label='After denoising', density=True)
    ax.set_yscale('log')
    ax.axvline(0.20, color='black', ls='--', lw=0.8, alpha=0.6)
    ax.axvline(-0.20, color='black', ls='--', lw=0.8, alpha=0.6)
    ax.set_ylim(bottom=1e-3)
    ax.set_xlim(-1.0, 1.5)
    ax.set_xlabel(r'$\Delta E/E$', fontsize=10.5)
    ax.set_ylabel('Density (log scale)', fontsize=10.5)

    txt = f'{e_kev:g} keV, {noise_str}'
    if row_n is not None and row_d is not None:
        txt += (f"\nskew: {row_n['skew']:.2f} → {row_d['skew']:.2f}"
                f"\nkurt: {row_n['kurt']:.2f} → {row_d['kurt']:.2f}"
                f"\nout>20%: {row_n['outlier_frac_20']*100:.1f}% → {row_d['outlier_frac_20']*100:.1f}%")
    ax.set_title(txt, fontsize=9.5, loc='left')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3, which='both')


def plot_lowE_tails(fit_dir, rows, energies, noise_tag, noise_str, output_path):
    fig, axes = plt.subplots(1, len(energies), figsize=(6.2 * len(energies), 5))
    if len(energies) == 1:
        axes = [axes]
    for ax, e in zip(axes, energies):
        path = os.path.join(fit_dir, f'fit_efficiency_{int(e)}keV.h5')
        e_kev, de_n, de_d = _load_delta_e(path)
        row_n = _row(rows, float(e), noise_tag, 'noisy')
        row_d = _row(rows, float(e), noise_tag, 'denoised')
        _hist_panel(ax, de_n, de_d, e_kev, noise_str, row_n, row_d)
    fig.suptitle('Low energy: "core narrows, tail survives" — denoising improves the\n'
                 'fixed-threshold pass rate while excess kurtosis (tail weight) gets worse',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, output_path)


def plot_highE_contrast(fit_dir, rows, energies, noise_tag, noise_str, output_path):
    fig, axes = plt.subplots(1, len(energies), figsize=(6.2 * len(energies), 5))
    if len(energies) == 1:
        axes = [axes]
    for ax, e in zip(axes, energies):
        path = os.path.join(fit_dir, f'fit_efficiency_{int(e)}keV.h5')
        e_kev, de_n, de_d = _load_delta_e(path)
        row_n = _row(rows, float(e), noise_tag, 'noisy')
        row_d = _row(rows, float(e), noise_tag, 'denoised')
        _hist_panel(ax, de_n, de_d, e_kev, noise_str, row_n, row_d)
        ax.set_xlim(-0.3, 0.3)
    fig.suptitle('High energy contrast: clean, near-Gaussian, no residual tail\n'
                 '— resolution and pass-rate agree here', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, output_path)


def plot_trends(rows, output_path):
    metrics = [
        ('skew', r'$|{\rm skewness}|$', True),
        ('kurt', r'$|{\rm excess\ kurtosis}|$', True),
        ('outlier_frac_20', 'Outlier fraction ($|\\Delta E/E|$>20%)', False),
        ('chi2ndf', r'$\chi^2$/ndf (Gaussian fit)', False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, (col, label, absval) in zip(axes.flat, metrics):
        for noise_tag, noise_str, ls in [('1p2mV', '1.2 mV', '-'), ('7p5mV', '7.5 mV', '--')]:
            for curve, color in [('noisy', COLOR_NOISY), ('denoised', COLOR_DENOISED)]:
                sub = sorted((r for r in rows if r['noise'] == noise_tag and r['curve'] == curve),
                             key=lambda r: r['energy_kev'])
                x = [r['energy_kev'] for r in sub]
                y = [abs(r[col]) if absval else r[col] for r in sub]
                ax.plot(x, y, marker='o', ms=4, lw=1.3, ls=ls, color=color,
                        label=f'{curve}, {noise_str}')
        ax.set_xscale('log')
        if col in ('kurt', 'chi2ndf'):
            ax.set_yscale('log')
        ax.set_xlabel('True energy (keV)', fontsize=9.5)
        ax.set_ylabel(label, fontsize=9.5)
        ax.grid(alpha=0.3, which='both')
        ax.tick_params(labelsize=8.5)
    axes.flat[0].legend(fontsize=7.5, loc='upper right', ncol=1)
    fig.suptitle('Non-Gaussianity metrics vs energy — all four worsen monotonically\n'
                 'as SNR drops (Spearman ρ<0, p<0.02 for all 4 combinations, see summary report)',
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, output_path)


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit_dir_1p2mV', required=True)
    ap.add_argument('--fit_dir_7p5mV', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--output_dir', default='results/analysis')
    args = ap.parse_args()

    rows = _load_csv(args.csv)

    plot_lowE_tails(args.fit_dir_1p2mV, rows, [1, 5], '1p2mV', '1.2 mV',
                     os.path.join(args.output_dir, 'shape_diagnostic_illustration_1_lowE_tails.png'))
    plot_highE_contrast(args.fit_dir_1p2mV, rows, [239, 511], '1p2mV', '1.2 mV',
                         os.path.join(args.output_dir, 'shape_diagnostic_illustration_2_highE_contrast.png'))
    plot_trends(rows, os.path.join(args.output_dir, 'shape_diagnostic_illustration_3_trends.png'))


if __name__ == '__main__':
    main()
