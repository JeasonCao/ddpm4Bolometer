"""Overlay two denoised test sets on the QA example windows.

Both HDF5s must come from ``scripts/denoise_test_set.py`` on the same
clean/noise test pair (same window order). Examples are the ones of
``scripts/plot_triexp_inference.py`` (typical-energy singles + pileups).

Each row: time domain (noisy / clean / A / B), residuals (A - clean,
B - clean), residual PSD (noise / clean / both residuals).

Usage:
    python3 -u scripts/compare_denoised_examples.py \\
        --h5_a .../denoised_test_000.h5 --label_a ep95 \\
        --h5_b .../denoised_test_000_ep190.h5 --label_b ep190 \\
        --out_prefix .../qa_compare_ep95_ep190/examples
"""
import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.plot_triexp_inference import (
    TYPICAL_E_KEV, PILEUP_DT_S, compute_psd, pick_single, pick_pileup)


def plot_rows(fa, fb, idxs, titles, la, lb, out_png):
    fs = float(fa.attrs['f_sample'])
    n = len(idxs)
    fig, axes = plt.subplots(n, 3, figsize=(17, 2.6 * n), squeeze=False)
    rms = lambda a: np.sqrt(np.mean(a ** 2))
    for r, (i, title) in enumerate(zip(idxs, titles)):
        c = fa['waveforms_clean'][i].astype(np.float64) * 1e3
        y = fa['waveforms_noisy'][i].astype(np.float64) * 1e3
        a = fa['waveforms'][i].astype(np.float64) * 1e3
        b = fb['waveforms'][i].astype(np.float64) * 1e3
        t = np.arange(c.size) / fs
        ax_t, ax_r, ax_p = axes[r]
        ax_t.plot(t, y, color='tab:red', lw=0.4, alpha=0.5, label='noisy')
        ax_t.plot(t, c, color='tab:blue', lw=1.2, label='clean')
        ax_t.plot(t, a, color='tab:green', lw=0.8, ls='--', label=la)
        ax_t.plot(t, b, color='tab:purple', lw=0.8, ls=':', label=lb)
        ax_t.set_ylabel('amplitude (mV)')
        ax_t.set_title(title, fontsize=9)

        ax_r.plot(t, a - c, color='tab:green', lw=0.6, label=la)
        ax_r.plot(t, b - c, color='tab:purple', lw=0.6, label=lb)
        ax_r.axhline(0, color='k', lw=0.5)
        ax_r.set_ylabel('denoised - clean (mV)')
        ax_r.set_title(f'residual RMS: {la} {rms(a - c):.3f}, '
                       f'{lb} {rms(b - c):.3f} mV', fontsize=9)

        for x, color, lw, label in ((y - c, 'tab:gray', 0.8, 'noise'),
                                    (c, 'tab:blue', 1.0, 'clean'),
                                    (a - c, 'tab:green', 0.8, f'residual {la}'),
                                    (b - c, 'tab:purple', 0.8,
                                     f'residual {lb}')):
            f, p = compute_psd(x, fs)
            ax_p.loglog(f, p, color=color, lw=lw, label=label, alpha=0.85)
        ax_p.set_ylabel('PSD (mV$^2$/Hz)')
        ax_p.grid(True, which='both', ls=':', lw=0.3, alpha=0.5)
        for ax in (ax_t, ax_r):
            ax.grid(True, alpha=0.3)
        if r == 0:
            for ax, loc in ((ax_t, 'upper right'), (ax_r, 'upper right'),
                            (ax_p, 'lower left')):
                ax.legend(fontsize=7, loc=loc, frameon=False)
    for ax in axes[-1, :2]:
        ax.set_xlabel('time (s)')
    axes[-1, 2].set_xlabel('frequency (Hz)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5_a', required=True)
    p.add_argument('--h5_b', required=True)
    p.add_argument('--label_a', default='A')
    p.add_argument('--label_b', default='B')
    p.add_argument('--out_prefix', required=True,
                   help='Writes <prefix>_single.png and <prefix>_pileup.png')
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    with h5py.File(args.h5_a, 'r') as fa, h5py.File(args.h5_b, 'r') as fb:
        E1 = fa['energies_1'][:]
        E2 = np.nan_to_num(fa['energies_2'][:])
        is_pu = fa['is_pileup'][:].astype(bool)
        dt = fa['onsets_2'][:] - fa['onsets_1'][:]
        idx_s = pick_single(E1, is_pu, TYPICAL_E_KEV)
        idx_p = pick_pileup(E1, E2, dt, is_pu, PILEUP_DT_S)
        plot_rows(fa, fb, idx_s,
                  [f'single, E = {E1[i]:.0f} keV (window {i})' for i in idx_s],
                  args.label_a, args.label_b, f'{args.out_prefix}_single.png')
        plot_rows(fa, fb, idx_p,
                  [f'pileup, E1 = {E1[i]:.0f}, E2 = {E2[i]:.0f} keV, '
                   f'$\\Delta t$ = {dt[i] * 1e3:.0f} ms (window {i})'
                   for i in idx_p],
                  args.label_a, args.label_b, f'{args.out_prefix}_pileup.png')


if __name__ == '__main__':
    main()
