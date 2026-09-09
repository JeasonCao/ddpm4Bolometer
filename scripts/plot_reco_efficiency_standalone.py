"""
Standalone reconstruction-efficiency-vs-energy plot (no trigger overlay),
for paper Sec. 4.4.

This is deliberately a thin wrapper around plot_combined_efficiency.py's
already-validated logistic / double-logistic fitting (src/basics/plot_utils),
so the reported threshold-crossing energies stay consistent with the
combined figure and with the numbers already written into the paper draft.
Do NOT use plot_efficiency_threshold.py's poly-4 fit for this purpose —
it disagrees with the logistic fit when data points near the crossing are
sparse (see 2026-08-13 session notes).

Usage
-----
python scripts/plot_reco_efficiency_standalone.py \
    --reco_json_1p2mV results/plots/efficiency/json \
    --reco_json_7p5mV results/plots/efficiency_7p5mV_ext511/json \
    --output_dir results/plots/efficiency \
    --threshold 90 --max_energy 511
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.basics.plot_utils import (logistic4, fit_efficiency_curve,
                                   double_logistic, fit_efficiency_curve_double)
from scripts.plot_combined_efficiency import _load_reco, _NOISE_LABELS

def plot_reco_only(noise_tag, reco_data, output_path, threshold_pct=90.0, x_max=511.0):
    noise_str = _NOISE_LABELS.get(noise_tag, noise_tag)
    e_r, eff_b, eff_a, cut = reco_data
    cut_pct = int(round(cut * 100))

    fig, ax = plt.subplots(figsize=(8, 6))
    xlim = (1.0 * 0.7, x_max * 1.4)
    crossings = []

    def _fit_and_plot(energies, eff, color, label, model):
        if model == 'double':
            popt, le_min, le_max, thr = fit_efficiency_curve_double(energies, eff, threshold_pct)
            curve_fn = double_logistic
        else:
            popt, le_min, le_max, thr = fit_efficiency_curve(energies, eff, threshold_pct)
            curve_fn = logistic4
        log_e_plot = np.linspace(le_min, le_max, 500)
        ax.plot(np.exp(log_e_plot), curve_fn(log_e_plot, *popt),
               color=color, lw=2.0, alpha=0.85)
        ax.scatter(energies, eff, color=color, marker='o', s=55,
                  edgecolors='white', linewidth=0.6, zorder=5, label=label)
        if thr is not None:
            crossings.append((label, color, thr))

    before_model = 'double' if noise_tag == '7p5mV' else 'single'
    _fit_and_plot(e_r, eff_b, 'steelblue', f'Before denoising (|ΔE/E|<{cut_pct}%)', before_model)
    _fit_and_plot(e_r, eff_a, 'tomato',    f'After denoising (|ΔE/E|<{cut_pct}%)', 'single')

    ax.axhline(threshold_pct, color='black', ls=':', lw=1.3, label=f'{threshold_pct:.0f}% threshold')
    for label, color, thr in crossings:
        ax.axvline(thr, color=color, ls=':', lw=1.0, alpha=0.6)
        ax.annotate(f'{thr:.1f} keV', xy=(thr, 0), xycoords=('data', 'axes fraction'),
                   xytext=(0, -4), textcoords='offset points',
                   ha='center', va='top', fontsize=8, color=color, rotation=90,
                   annotation_clip=False)

    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(-5, 112)
    ax.set_xlabel('Energy (keV)', fontsize=11)
    ax.set_ylabel('Efficiency (%)', fontsize=11)
    ax.set_title(f'Reconstruction Efficiency vs Energy\nTri-exp fit | noise RMS = {noise_str}', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {output_path}')
    for label, color, thr in crossings:
        print(f'  {label}: {thr:.2f} keV')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reco_json_1p2mV', default=os.path.join(_RESULTS_DIR, 'plots', 'efficiency', 'json'))
    ap.add_argument('--reco_json_7p5mV', default=os.path.join(_RESULTS_DIR, 'plots', 'efficiency_7p5mV_ext511', 'json'))
    ap.add_argument('--output_dir', default=os.path.join(_RESULTS_DIR, 'plots', 'efficiency'))
    ap.add_argument('--threshold', type=float, default=90.0)
    ap.add_argument('--max_energy', type=float, default=511.0)
    args = ap.parse_args()

    dirs = {'1p2mV': args.reco_json_1p2mV, '7p5mV': args.reco_json_7p5mV}
    for noise_tag, jdir in dirs.items():
        reco_data = _load_reco(jdir, noise_tag, args.max_energy, method='triexp')
        if reco_data is None:
            print(f'No data for {noise_tag}')
            continue
        out = os.path.join(args.output_dir, f'reco_efficiency_standalone_{noise_tag}.png')
        plot_reco_only(noise_tag, reco_data, out, threshold_pct=args.threshold, x_max=args.max_energy)


if __name__ == '__main__':
    main()
