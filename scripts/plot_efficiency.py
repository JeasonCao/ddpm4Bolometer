"""
Plot small-signal reconstruction efficiency: before vs after DDPM denoising.

Processes all fit files in a directory.  Multi-point energy calibration
(E [keV] = k·amp + b, separate for noisy/denoised × tri-exp/max-baseline)
is used to convert amplitudes to energy before computing efficiency metrics.
(Historical note: before the 2026-09-09 run_fitting.py baseline-bug fix, the
tri-exp fit carried a spurious ~5.3-5.9% systematic bias, absorbed by the
calibration fit; the true intrinsic bias is ~0.02-0.04%, see diary. The
energy-domain cut is set to the same value as the max-baseline cut either way.)

Layout (one figure per energy point)
------
Row 1 — 4 bar charts:
  [0] Tri-exp convergence rate
        converged = fr.success AND amp_min < A < amp_max AND chi²/ndf < chi2_max
  [1] Waveform–template Pearson correlation match rate
        r = cov(wf, tmpl) / (σ_wf · σ_tmpl), range [−1, 1]; threshold r > corr_min
        shape similarity metric (amplitude-invariant)
  [2] Tri-exp energy efficiency
        |ΔE / E_true| < energy_cut_triexp  (NaN fit failures count as inefficient)
        systematic bias absorbed by energy calibration; same cut as max-baseline
  [3] Max-baseline energy efficiency
        |ΔE / E_true| < energy_cut_maxbase  (always finite, no fitter bias)

Row 2 — 2 histograms:
  [0] Tri-exp (E_reco − E_true) / E_true distribution  (signed, before/after)
  [1] Max-baseline signed energy-deviation distribution
  Both show ± cut threshold lines so efficiency fraction is visually obvious.

A JSON summary is saved alongside each PNG.

Usage
-----
python scripts/plot_efficiency.py \\
    --fit_dir   results/fit_results/efficiency_1p2mV \\
    --model_cfg /mnt/d/storage/research/DDPM4bolometer/model_output_cloud
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import glob
import h5py
import numpy as np
from src.basics.plot_utils import suffix_from_model_dir
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_ENERGY_CUT_TRIEXP  = 0.05   # 5%: bias absorbed by energy calibration
DEFAULT_ENERGY_CUT_MAXBASE = 0.05   # 5%: unbiased estimator
DEFAULT_CORR_MIN           = 0.90


# ── bar helpers ───────────────────────────────────────────────────────────────

def bar_pair(ax, val_n, val_d, title, ylabel='Rate (%)'):
    """Two-bar chart: before / after."""
    x, w = np.array([0.0]), 0.3
    b_n = ax.bar(x - w/2, [val_n], w, color='steelblue', alpha=0.85, label='Before')
    b_d = ax.bar(x + w/2, [val_d], w, color='tomato',    alpha=0.85, label='After')
    for bar in [b_n[0], b_d[0]]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, wrap=True)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)




# ── calibration ───────────────────────────────────────────────────────────────

def compute_calibrations(fit_dir):
    """Compute energy calibrations, shared with plot_reconstruction_quality.py
    (2026-09-10 unification — this used to run its own independent
    fit_energy_calibration() over all files <=max_energy, unweighted, free
    intercept, noisy and denoised each fit separately; that disagreed by
    ~1% with plot_reconstruction_quality.py's calibration for the same
    noise level, e.g. 2.20 vs 2.22 keV baseline resolution at 1.2mV).

    Now uses the same 5-point (239/511/727/1461/2615 keV) zero-intercept,
    inverse-variance-weighted regression from src/basics/plot_utils.py,
    fit once on DENOISED data and shared between noisy and denoised curves
    — mirrors how a real experiment calibrates once, on its best/final
    processing, and applies that single calibration to everything,
    including the before/after comparison. b is always 0.

    Returns the same 4-key dict shape as before ('triexp_n'/'triexp_d'/
    'maxbase_n'/'maxbase_d' -> (k, b)) so downstream code needs no changes.
    """
    from src.basics.plot_utils import fit_global_calibration_zero_intercept
    k_triexp,  _res_t = fit_global_calibration_zero_intercept(fit_dir, method='triexp')
    k_maxbase, _res_m = fit_global_calibration_zero_intercept(fit_dir, method='maxbase')
    return {
        'triexp_n':  (k_triexp,  0.0),
        'triexp_d':  (k_triexp,  0.0),
        'maxbase_n': (k_maxbase, 0.0),
        'maxbase_d': (k_maxbase, 0.0),
    }


# ── per-file plotting ─────────────────────────────────────────────────────────

def plot_efficiency(fit_file: str, output: str, calibs: dict,
                    energy_cut_triexp:  float = DEFAULT_ENERGY_CUT_TRIEXP,
                    energy_cut_maxbase: float = DEFAULT_ENERGY_CUT_MAXBASE,
                    corr_min:           float = DEFAULT_CORR_MIN):

    # ── load raw amplitudes ────────────────────────────────────────────────────
    with h5py.File(fit_file, 'r') as f:
        energy_kev   = float(f.attrs['energy_kev'])
        true_amp_V   = float(f.attrs['true_amp_V'])
        chi2_max     = float(f.attrs.get('chi2_max', float('nan')))
        amp_min      = float(f.attrs.get('amp_min',  float('nan')))
        amp_max_attr = float(f.attrs.get('amp_max',  float('nan')))
        snr_db         = f['snr_db'][:]
        conv_n         = f['converged_noisy'][:].astype(bool)
        conv_d         = f['converged_denoised'][:].astype(bool)
        amp_triexp_n   = f['amp_noisy'][:].astype(float)
        amp_triexp_d   = f['amp_denoised'][:].astype(float)
        amp_maxbase_n  = f['amp_maxbase_noisy'][:].astype(float)
        amp_maxbase_d  = f['amp_maxbase_denoised'][:].astype(float)
        fit_pearsonr_n = f['fit_pearsonr_noisy'][:].astype(float)
        fit_pearsonr_d = f['fit_pearsonr_denoised'][:].astype(float)

    n_total  = len(snr_db)
    snr_mean = float(np.nanmean(snr_db))
    ds_name  = os.path.basename(os.path.dirname(fit_file))

    # ── apply energy calibration ───────────────────────────────────────────────
    k_tn, b_tn = calibs['triexp_n']
    k_td, b_td = calibs['triexp_d']
    k_mn, b_mn = calibs['maxbase_n']
    k_md, b_md = calibs['maxbase_d']

    E_triexp_n  = k_tn * amp_triexp_n  + b_tn
    E_triexp_d  = k_td * amp_triexp_d  + b_td
    E_maxbase_n = k_mn * amp_maxbase_n + b_mn
    E_maxbase_d = k_md * amp_maxbase_d + b_md

    # ── signed energy deviations ───────────────────────────────────────────────
    edev_triexp_n  = (E_triexp_n  - energy_kev) / energy_kev
    edev_triexp_d  = (E_triexp_d  - energy_kev) / energy_kev
    edev_maxbase_n = (E_maxbase_n - energy_kev) / energy_kev
    edev_maxbase_d = (E_maxbase_d - energy_kev) / energy_kev

    # ── rates ──────────────────────────────────────────────────────────────────
    conv_rate_n = conv_n.mean() * 100
    conv_rate_d = conv_d.mean() * 100

    # efficiency: |ΔE/E| < threshold; NaN tri-exp failures count as inefficient
    eff_triexp_n = np.mean(np.where(np.isfinite(edev_triexp_n),
                                    np.abs(edev_triexp_n) < energy_cut_triexp, False)) * 100
    eff_triexp_d = np.mean(np.where(np.isfinite(edev_triexp_d),
                                    np.abs(edev_triexp_d) < energy_cut_triexp, False)) * 100
    eff_maxbase_n = np.mean(np.abs(edev_maxbase_n) < energy_cut_maxbase) * 100
    eff_maxbase_d = np.mean(np.abs(edev_maxbase_d) < energy_cut_maxbase) * 100

    # ── Pearson r: fit model curve vs clean template (computed at fit time) ───
    # NaN = fit failed → counts as not matched
    match_rate_n = float(np.mean(np.where(np.isfinite(fit_pearsonr_n),
                                          fit_pearsonr_n > corr_min, False)) * 100)
    match_rate_d = float(np.mean(np.where(np.isfinite(fit_pearsonr_d),
                                          fit_pearsonr_d > corr_min, False)) * 100)

    # ── figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 9))
    gs  = fig.add_gridspec(2, 4)

    ax_top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_hist_triexp  = fig.add_subplot(gs[1, 0:2])
    ax_hist_maxbase = fig.add_subplot(gs[1, 2:4])

    fig.suptitle(f'Efficiency Analysis — {energy_kev:.0f} keV  '
                 f'(SNR ≈ {snr_mean:.1f} dB,  N={n_total})',
                 fontsize=13)

    # row 1 — bar charts ────────────────────────────────────────────────────────
    conv_title = (
        f'Tri-exp convergence\n'
        f'fr.success  &  {amp_min:.0e} < A < {amp_max_attr:.1f} V\n'
        f'&  χ²/ndf < {chi2_max:.0f}'
    )
    bar_pair(ax_top[0], conv_rate_n, conv_rate_d, conv_title)

    bar_pair(ax_top[1], match_rate_n, match_rate_d,
             f'Fit–template match\n'
             f'r(tri-exp fit, clean template) > {corr_min:.2f}')

    bar_pair(ax_top[2], eff_triexp_n, eff_triexp_d,
             f'Tri-exp energy efficiency\n'
             f'|ΔE| / E_true < {energy_cut_triexp:.0%}')

    bar_pair(ax_top[3], eff_maxbase_n, eff_maxbase_d,
             f'Max-baseline energy efficiency\n'
             f'|ΔE| / E_true < {energy_cut_maxbase:.0%}')

    # row 2 — signed energy-deviation histograms (no clipping; overflow annotated)
    for ax, edev_n_raw, edev_d_raw, lbl, cut in [
        (ax_hist_triexp,  edev_triexp_n,  edev_triexp_d,
         'Tri-exp fit',    energy_cut_triexp),
        (ax_hist_maxbase, edev_maxbase_n, edev_maxbase_d,
         'Max-baseline',   energy_cut_maxbase),
    ]:
        fin_n = edev_n_raw[np.isfinite(edev_n_raw)]
        fin_d = edev_d_raw[np.isfinite(edev_d_raw)]
        nan_n = int(np.sum(~np.isfinite(edev_n_raw)))
        nan_d = int(np.sum(~np.isfinite(edev_d_raw)))

        # Display range: ±1.0 or just wide enough to show the cut + some margin
        display_range = max(1.0, cut * 5)
        bins = np.linspace(-display_range, display_range, 80)

        # Count events outside the displayed range (not clipped, just not shown)
        out_n = int(np.sum(np.abs(fin_n) > display_range))
        out_d = int(np.sum(np.abs(fin_d) > display_range))

        shown_n = fin_n[np.abs(fin_n) <= display_range]
        shown_d = fin_d[np.abs(fin_d) <= display_range]

        ax.hist(shown_n, bins=bins, density=True, alpha=0.55, color='steelblue',
                label=f'Before  (shown={len(shown_n)}, outside={out_n}, NaN={nan_n})')
        ax.hist(shown_d, bins=bins, density=True, alpha=0.55, color='tomato',
                label=f'After   (shown={len(shown_d)}, outside={out_d}, NaN={nan_d})')
        ax.axvline(-cut, color='black', ls='--', lw=1.2, label=f'±{cut:.0%} cut')
        ax.axvline( cut, color='black', ls='--', lw=1.2)
        ax.axvline(0,    color='gray',  ls=':',  lw=0.8)
        ax.set_xlim(-display_range, display_range)
        ax.set_xlabel('(E_reco − E_true) / E_true', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'Signed energy deviation — {lbl}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.08,
                        hspace=0.55, wspace=0.35)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()

    # ── save JSON summary ──────────────────────────────────────────────────────
    summary = {
        'energy_kev':            energy_kev,
        'dataset':               ds_name,
        'n_total':               n_total,
        'snr_mean_db':           round(snr_mean, 2),
        'conv_rate_before':      round(conv_rate_n,    2),
        'conv_rate_after':       round(conv_rate_d,    2),
        'fit_template_match_before': round(match_rate_n, 2),
        'fit_template_match_after':  round(match_rate_d, 2),
        'eff_triexp_before':     round(eff_triexp_n,   2),
        'eff_triexp_after':      round(eff_triexp_d,   2),
        'eff_maxbase_before':    round(eff_maxbase_n,  2),
        'eff_maxbase_after':     round(eff_maxbase_d,  2),
        'energy_cut_triexp':     energy_cut_triexp,
        'energy_cut_maxbase':    energy_cut_maxbase,
        'corr_min':              corr_min,
        'note_pearsonr':         'r(tri-exp fitted curve, clean template); NaN = fit failed',
        'chi2_max':              chi2_max,
        'amp_min_V':             amp_min,
        'amp_max_V':             amp_max_attr,
        'true_amp_V':            true_amp_V,
        'calib_triexp_n':        {'k': round(k_tn, 3), 'b': round(b_tn, 3)},
        'calib_triexp_d':        {'k': round(k_td, 3), 'b': round(b_td, 3)},
        'calib_maxbase_n':       {'k': round(k_mn, 3), 'b': round(b_mn, 3)},
        'calib_maxbase_d':       {'k': round(k_md, 3), 'b': round(b_md, 3)},
    }
    json_dir = os.path.join(os.path.dirname(os.path.abspath(output)), 'json')
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, os.path.basename(output).replace('.png', '.json'))
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2)

    print(f"Saved → {output}")
    print(f"  conv: {conv_rate_n:.1f}%→{conv_rate_d:.1f}%  "
          f"| triexp({energy_cut_triexp:.0%}): {eff_triexp_n:.1f}%→{eff_triexp_d:.1f}%  "
          f"| maxbase({energy_cut_maxbase:.0%}): {eff_maxbase_n:.1f}%→{eff_maxbase_d:.1f}%"
          + (f"  | match(r>{corr_min:.2f}): {match_rate_n:.1f}%→{match_rate_d:.1f}%"
             if match_rate_n is not None else ''))


# ── batch runner ──────────────────────────────────────────────────────────────

def run(fit_dir, out_dir, suffix, energy_cut_triexp, energy_cut_maxbase, corr_min,
       max_energy=None):
    fit_files = sorted(glob.glob(os.path.join(fit_dir, 'fit_efficiency_*.h5')))
    if not fit_files:
        print(f"No fit files found in {fit_dir}")
        return

    if max_energy is not None:
        kept = []
        for fp in fit_files:
            with h5py.File(fp, 'r') as f:
                e_kev = float(f.attrs['energy_kev'])
            if e_kev <= max_energy:
                kept.append(fp)
        fit_files = kept
        if not fit_files:
            print(f"No fit files with energy_kev <= {max_energy}")
            return

    print(f"Computing energy calibrations (5-point zero-intercept, shared with "
          f"plot_reconstruction_quality.py) from {fit_dir} …")
    calibs = compute_calibrations(fit_dir)
    for tag, (k, b) in calibs.items():
        print(f"  {tag:12s}: k={k:.2f} keV/V, b={b:.2f} keV")

    ds_tag   = os.path.basename(fit_dir)           # e.g. "efficiency_1p2mV"
    ds_short = ds_tag.replace('efficiency_', '')   # e.g. "1p2mV"
    t_tag    = f'_t{int(energy_cut_triexp  * 100)}pct'
    m_tag    = f'_m{int(energy_cut_maxbase * 100)}pct'

    for fit_file in fit_files:
        stem       = os.path.splitext(os.path.basename(fit_file))[0]
        energy_tag = stem[len('fit_efficiency_'):]  # e.g. "5keV"
        output = os.path.join(
            out_dir,
            f'efficiency_{energy_tag}_{ds_short}{suffix}{t_tag}{m_tag}.png')
        plot_efficiency(fit_file, output, calibs,
                        energy_cut_triexp=energy_cut_triexp,
                        energy_cut_maxbase=energy_cut_maxbase,
                        corr_min=corr_min)


def main():
    parser = argparse.ArgumentParser(
        description='Plot small-signal reconstruction efficiency (energy-calibrated)')
    parser.add_argument('--fit_dir',
                        required=True,
                        help='Directory of fit HDF5 files (fit_efficiency_*.h5)')
    parser.add_argument('--energy_cut_triexp',  type=float,
                        default=DEFAULT_ENERGY_CUT_TRIEXP,
                        help=f'Tri-exp energy efficiency cut (default {DEFAULT_ENERGY_CUT_TRIEXP:.0%})')
    parser.add_argument('--energy_cut_maxbase', type=float,
                        default=DEFAULT_ENERGY_CUT_MAXBASE,
                        help=f'Max-baseline energy cut (default {DEFAULT_ENERGY_CUT_MAXBASE:.0%})')
    parser.add_argument('--corr_min',    type=float, default=DEFAULT_CORR_MIN)
    parser.add_argument('--output_dir',  default=None)
    parser.add_argument('--suffix',      default='')
    parser.add_argument('--model_cfg',   default=None, metavar='MODEL_DIR')
    parser.add_argument('--max_energy',  type=float, default=None,
                        help='Only use/plot files with energy_kev <= this value '
                             '(also restricts the calibration fit)')
    args = parser.parse_args()

    suffix  = suffix_from_model_dir(args.model_cfg) if args.model_cfg else args.suffix
    out_dir = args.output_dir or os.path.join(_RESULTS_DIR, 'plots', 'efficiency')

    run(args.fit_dir, out_dir, suffix,
        args.energy_cut_triexp, args.energy_cut_maxbase, args.corr_min,
        max_energy=args.max_energy)


if __name__ == '__main__':
    main()
