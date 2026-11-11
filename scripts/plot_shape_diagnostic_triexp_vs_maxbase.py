"""
2026-09-10: comparison figure for the shape-diagnostic v2 finding — the
"core narrows but a stubborn non-Gaussian tail survives after denoising"
pattern shows up in BOTH the tri-exp fit amplitude AND the max-baseline
amplitude, at the same low-energy point. Since max-baseline has no
nonlinear fitter at all, this argues the effect originates in the
denoising model itself, not in tri-exp fit parameter degeneracy.

See results/analysis/reco_efficiency_shape_diagnostics_v2_summary_20260910.md
for the full quantitative writeup this figure illustrates.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIT_FILE = os.path.join(_RESULTS_DIR, 'fit_results', 'efficiency_1p2mV', 'fit_efficiency_1keV.h5')
OUT = os.path.join(_RESULTS_DIR, 'analysis', 'shape_diagnostic_illustration_4_triexp_vs_maxbase.png')


def _panel(ax, de_n, de_d, title):
    bins = np.linspace(-1.0, 1.5, 120)
    ax.hist(de_n, bins=bins, histtype='step', color='steelblue', lw=1.4,
            label='Before denoising', density=True)
    ax.hist(de_d, bins=bins, histtype='step', color='tomato', lw=1.4,
            label='After denoising', density=True)
    ax.set_yscale('log')
    ax.axvline(0.20, color='black', ls='--', lw=0.8, alpha=0.6)
    ax.axvline(-0.20, color='black', ls='--', lw=0.8, alpha=0.6)
    ax.set_ylim(bottom=1e-3)
    ax.set_xlim(-1.0, 1.5)
    ax.set_xlabel(r'$\Delta E/E$', fontsize=10.5)
    ax.set_ylabel('Density (log scale)', fontsize=10.5)
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.grid(alpha=0.3, which='both')


def main():
    with h5py.File(FIT_FILE) as f:
        true_amp = float(f.attrs['true_amp_V'])
        e_kev = float(f.attrs['energy_kev'])
        succ_n = f['success_noisy'][:].astype(bool)
        succ_d = f['success_denoised'][:].astype(bool)
        de_tri_n = f['amp_noisy'][:][succ_n] / true_amp - 1.0
        de_tri_d = f['amp_denoised'][:][succ_d] / true_amp - 1.0
        de_mb_n  = f['amp_maxbase_noisy'][:] / true_amp - 1.0
        de_mb_d  = f['amp_maxbase_denoised'][:] / true_amp - 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _panel(axes[0], de_tri_n, de_tri_d, f'Tri-exp fit amplitude — {e_kev:g} keV, 1.2 mV')
    _panel(axes[1], de_mb_n,  de_mb_d,  f'Max-baseline amplitude — {e_kev:g} keV, 1.2 mV\n(no nonlinear fitter involved)')
    fig.suptitle('Core-narrows-but-tail-survives: present in both amplitude estimators\n'
                 '(argues the effect originates in the denoiser, not in tri-exp fit degeneracy)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f'Saved -> {OUT}')


if __name__ == '__main__':
    main()
