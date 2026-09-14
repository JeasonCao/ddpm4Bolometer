"""
Single-panel paper figure: illustrates the reconstruction-efficiency cut
definition (|ΔE/E| < cut) on the denoised ΔE/E distribution for one
representative energy point.

This is a clean, publication-style extraction of the same ΔE/E-vs-cut idea
already computed internally by plot_efficiency.py (its "Signed energy
deviation" panel), which is a busy 6-panel QA figure not meant for a paper.
This script draws just the one distribution, shades pass/fail regions, and
annotates the resulting efficiency fraction.

Calibration: single-point (k = energy_kev / true_amp_V), matching the
template-based per-point convention used elsewhere for illustrative figures
(e.g. plot_single_energy_example.py). Not the multi-point global calibration
used for the quantitative efficiency-vs-energy curves — this is a schematic,
not a quantitative result.

Usage
-----
python scripts/plot_efficiency_cut_illustration.py \
    --fit_file results/fit_results/efficiency_7p5mV/fit_efficiency_15keV.h5 \
    --cut 0.05 \
    --output results/plots/reconstruction_quality/efficiency_cut_illustration_15keV_7p5mV.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import re
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_NOISE_RMS_V = {'1p2mV': 0.0012, '7p5mV': 0.0075}
# NOTE 2026-09-10: kept in sync by hand with the canonical baseline-resolution
# numbers established in the 2026-09-10 diary (zero-intercept, unified
# calibration). Update both together if the calibration changes again.
_BASELINE_RES_KEV = {'1p2mV': 2.21, '7p5mV': 13.81}


def _noise_tag_from_path(path):
    """Best-effort extraction of the '1p2mV'/'7p5mV' tag from the fit_file
    path (directory name or filename), so the figure can label which noise
    condition it shows without relying on the caller to pass it separately."""
    m = re.search(r'(\d+p\d+mV)', path)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit_file', required=True)
    ap.add_argument('--cut', type=float, default=0.05,
                    help='Efficiency cut on |ΔE/E| (default 0.05 = 5%%)')
    ap.add_argument('--n_bins', type=int, default=60)
    ap.add_argument('--method', choices=['triexp', 'maxbase'], default='triexp',
                    help='Amplitude estimator: tri-exp fit (default) or max-baseline')
    ap.add_argument('--output', default=os.path.join(
        _RESULTS_DIR, 'plots', 'reconstruction_quality',
        'efficiency_cut_illustration.png'))
    args = ap.parse_args()

    amp_key = 'amp_denoised' if args.method == 'triexp' else 'amp_maxbase_denoised'
    with h5py.File(args.fit_file, 'r') as f:
        energy_kev = float(f.attrs['energy_kev'])
        true_amp_V = float(f.attrs['true_amp_V'])
        amp = f[amp_key][:]
        n_total = int(f.attrs['n_events'])
        # max-baseline has no fitter to fail to converge -- it's a plain
        # max()/median() read, always defined (verified 2026-09-10: no
        # NaN/inf in amp_maxbase_* across the dataset). Only tri-exp needs
        # the converged-event mask.
        if args.method == 'triexp':
            conv = f['converged_denoised'][:]
        else:
            conv = np.ones(n_total, dtype=bool)

    noise_tag = _noise_tag_from_path(args.fit_file)
    noise_label = {'1p2mV': '1.2 mV', '7p5mV': '7.5 mV'}.get(noise_tag, noise_tag or 'unknown noise')
    res_kev = _BASELINE_RES_KEV.get(noise_tag)
    noise_str = f'{noise_label} (baseline resolution ≈ {res_kev:.2f} keV)' if res_kev else noise_label

    amp = amp[conv]
    n_conv = len(amp)
    n_nonconv = n_total - n_conv

    k = energy_kev / true_amp_V  # keV/V, single-point template calibration
    e_reco = amp * k
    dE_E = (e_reco - energy_kev) / energy_kev  # signed relative deviation

    n_pass = int((np.abs(dE_E) < args.cut).sum())
    efficiency_pct = 100.0 * n_pass / n_total  # denominator = all events, not just converged
    efficiency_pct_of_converged = 100.0 * n_pass / n_conv

    # ── plot ─────────────────────────────────────────────────────────────────
    lo = min(dE_E.min(), -args.cut * 1.6)
    hi = max(dE_E.max(), args.cut * 1.6)
    bins = np.linspace(lo, hi, args.n_bins)

    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    counts, edges = np.histogram(dE_E, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    pass_mask = np.abs(centers) < args.cut
    ax.bar(centers[pass_mask], counts[pass_mask], width=width,
           color='tomato', alpha=0.85, edgecolor='white', linewidth=0.3,
           label=f'Pass  (|ΔE/E| < {args.cut:.0%})')
    ax.bar(centers[~pass_mask], counts[~pass_mask], width=width,
           color='steelblue', alpha=0.35, edgecolor='white', linewidth=0.3,
           label=f'Fail  (|ΔE/E| ≥ {args.cut:.0%})')

    ax.axvline(-args.cut, color='black', ls='--', lw=1.3)
    ax.axvline(args.cut, color='black', ls='--', lw=1.3,
               label=f'±{args.cut:.0%} cut boundary')
    ax.axvline(0.0, color='gray', ls=':', lw=1.0, alpha=0.7)

    ax.set_xlabel(r'$\Delta E / E = (E_\mathrm{reco} - E_\mathrm{true}) / E_\mathrm{true}$',
                  fontsize=11)
    ax.set_ylabel('Counts', fontsize=11)
    method_label = 'tri-exp fit' if args.method == 'triexp' else 'max-baseline'
    ax.set_title(
        f'Reconstruction-efficiency cut illustration — {energy_kev:.0f} keV\n'
        f'{noise_str}, denoised, {method_label}', fontsize=10.5)

    # NOTE 2026-09-10: the histogram only has ΔE/E defined for CONVERGED
    # events (n_conv) — non-converged fits (n_nonconv) have no amplitude
    # estimate at all and can't be placed on this axis, so they're silently
    # absent from the bars. But the headline efficiency is computed against
    # n_total (all events), so a low convergence rate makes the annotated
    # percentage look inconsistent with "nearly everything shown passes".
    # Spell out the breakdown explicitly so the two numbers don't appear to
    # contradict each other. Max-baseline has no non-converged concept
    # (n_nonconv==0 always), so the breakdown collapses to the plain form.
    conv_pct = 100.0 * n_conv / n_total
    if n_nonconv == 0:
        annotation = f'Reconstruction efficiency = {n_pass} / {n_total} = {efficiency_pct:.1f}%'
    else:
        annotation = (
            f'Reconstruction efficiency = {n_pass} / {n_total} = {efficiency_pct:.1f}%\n'
            f'  converged: {n_conv}/{n_total} ({conv_pct:.1f}%) — only these appear in the histogram\n'
            f'  of converged, pass cut: {n_pass}/{n_conv} ({100.0*n_pass/n_conv:.1f}%)\n'
            f'  non-converged ({n_nonconv}, {100-conv_pct:.1f}%) count as fail, not shown above'
        )
    ax.text(0.03, 0.95, annotation,
            transform=ax.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.9))

    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f'Saved -> {args.output}')
    print(f'  N_total={n_total}  N_converged={n_conv}  N_pass={n_pass}')
    print(f'  efficiency (of all events)       = {efficiency_pct:.2f}%')
    print(f'  efficiency (of converged events) = {efficiency_pct_of_converged:.2f}%')


if __name__ == '__main__':
    main()
