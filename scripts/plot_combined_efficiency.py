"""
Combined reconstruction-efficiency + DT-trigger-efficiency plot, before vs
after DDPM denoising, on a single set of axes per noise level.

Reconstruction efficiency: |ΔE/E| < 7% on the tri-exp fitted amplitude
  (JSON from plot_efficiency.py).
Trigger efficiency: DT (derivative-threshold) trigger fires within the
  search window (JSON from run_dt_trigger.py).

Both series are fit with a 4-parameter logistic in log(energy) space
(src/basics/plot_utils.fit_efficiency_curve) — efficiency turn-on curves are
S-shaped and saturate at both ends, which a polynomial fits poorly.

Color encodes before(blue)/after(red); line style + marker shape encodes
reconstruction (solid, circle) vs trigger (dashed, triangle).

Usage
-----
python scripts/plot_combined_efficiency.py \
    --reco_json_1p2mV  results/plots/efficiency/json \
    --reco_json_7p5mV  results/plots/efficiency_7p5mV_ext511/json \
    --trig_json        results/plots/trigger/json \
    --output_dir        results/plots/combined \
    --threshold 90 --max_energy 511
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.basics.plot_utils import (logistic4, fit_efficiency_curve,
                                   double_logistic, fit_efficiency_curve_double,
                                   timestamped_path)

_NOISE_LABELS = {'1p2mV': '1.2 mV', '7p5mV': '7.5 mV'}


# ── data loaders ──────────────────────────────────────────────────────────────

def _load_reco(json_dir, noise_tag, max_energy, method='triexp'):
    """Reconstruction-efficiency (|ΔE/E|<cut) vs energy, single-suffix
    JSON files only (global-calibration files spanning a mismatched energy
    range are excluded — see earlier calibration-bias discussion).

    method: 'triexp' (tri-exp fitted amplitude) or 'maxbase' (max-baseline
    amplitude, unbiased estimator)."""
    files = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    records = []
    for jf in files:
        bn = os.path.basename(jf)
        if f'_{noise_tag}_' not in bn or bn.count(noise_tag) > 1:
            continue
        with open(jf) as f:
            records.append(json.load(f))
    records = [r for r in records if r['energy_kev'] <= max_energy]
    records.sort(key=lambda r: r['energy_kev'])
    if not records:
        return None
    eff_b_key  = f'eff_{method}_before'
    eff_a_key  = f'eff_{method}_after'
    cut_key    = f'energy_cut_{method}'
    return (np.array([r['energy_kev'] for r in records]),
           np.array([r[eff_b_key]    for r in records]),
           np.array([r[eff_a_key]    for r in records]),
           records[0][cut_key])


def _load_trigger(json_dir, noise_tag, max_energy):
    files = sorted(glob.glob(os.path.join(json_dir, f'trigger_*_{noise_tag}.json')))
    records = []
    for jf in files:
        with open(jf) as f:
            records.append(json.load(f))
    records = [r for r in records if r['energy_kev'] <= max_energy]
    records.sort(key=lambda r: r['energy_kev'])
    if not records:
        return None
    return (np.array([r['energy_kev']           for r in records]),
           np.array([r['trigger_eff_before']   for r in records]),
           np.array([r['trigger_eff_after']    for r in records]),
           records[0]['nsigma'], records[0]['n_above'])


# ── single combined figure ───────────────────────────────────────────────────

def plot_combined(noise_tag, reco_data, trig_data, output_path,
                  threshold_pct=90.0, x_max=511.0, method='triexp'):
    method_label = 'Tri-exp fit' if method == 'triexp' else 'Max-baseline'
    noise_str = _NOISE_LABELS.get(noise_tag, noise_tag)
    fig, ax = plt.subplots(figsize=(9, 6.5))

    xlim = (1.0 * 0.7, x_max * 1.4)
    crossings = []   # (label, color, energy)

    def _fit_and_plot(energies, eff, color, ls, marker, label, fill_marker=True, model='single'):
        if model == 'double':
            popt, le_min, le_max, thr = fit_efficiency_curve_double(energies, eff, threshold_pct)
            curve_fn = double_logistic
        else:
            popt, le_min, le_max, thr = fit_efficiency_curve(energies, eff, threshold_pct)
            curve_fn = logistic4
        log_e_plot = np.linspace(le_min, le_max, 500)
        ax.plot(np.exp(log_e_plot), curve_fn(log_e_plot, *popt),
               color=color, ls=ls, lw=2.0, alpha=0.85)
        ax.scatter(energies, eff, color=color, marker=marker, s=45 if fill_marker else 60,
                  facecolors=color if fill_marker else 'none',
                  edgecolors=color, linewidth=1.2, zorder=5, label=label)
        if thr is not None:
            crossings.append((label, color, thr))
        return thr

    if reco_data is not None:
        e_r, eff_r_b, eff_r_a, cut = reco_data
        cut_pct = int(round(cut * 100))
        # "before" reconstruction efficiency at higher noise can show a
        # two-stage shape (slow rise then an abrupt jump) that a single
        # logistic underfits in the middle — use the double-logistic there.
        before_model = 'double' if noise_tag == '7p5mV' else 'single'
        _fit_and_plot(e_r, eff_r_b, 'steelblue', '-', 'o',
                     f'{method_label}, before (|ΔE/E|<{cut_pct}%)', model=before_model)
        _fit_and_plot(e_r, eff_r_a, 'tomato', '-', 'o',
                     f'{method_label}, after (|ΔE/E|<{cut_pct}%)')

    if trig_data is not None:
        e_t, eff_t_b, eff_t_a, nsigma, n_above = trig_data
        _fit_and_plot(e_t, eff_t_b, 'steelblue', '--', '^',
                     'DT trigger, before', fill_marker=False)
        _fit_and_plot(e_t, eff_t_a, 'tomato', '--', '^',
                     'DT trigger, after', fill_marker=False)

    ax.axhline(threshold_pct, color='black', ls=':', lw=1.3,
              label=f'{threshold_pct:.0f}% threshold')

    for label, color, thr in crossings:
        ax.axvline(thr, color=color, ls=':', lw=1.0, alpha=0.6)
        ax.annotate(f'{thr:.1f} keV', xy=(thr, 0), xycoords=('data', 'axes fraction'),
                   xytext=(0, -4), textcoords='offset points',
                   ha='center', va='top', fontsize=7, color=color,
                   rotation=90, annotation_clip=False)

    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(-5, 112)
    ax.set_xlabel('Energy (keV)', fontsize=11)
    ax.set_ylabel('Efficiency (%)', fontsize=11)

    dt_label = ''
    if trig_data is not None:
        dt_label = f'  |  DT trigger: nsigma={trig_data[3]:.0f}, n_above={trig_data[4]} samples'
    ax.set_title(
        f'Reconstruction ({method_label}) & Trigger Efficiency vs Energy\n'
        f'noise RMS = {noise_str}{dt_label}',
        fontsize=11)
    ax.legend(fontsize=8.5, loc='lower right', ncol=1)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    output_path = timestamped_path(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {output_path}')
    for label, color, thr in crossings:
        print(f'  {label}: {thr:.2f} keV')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Combined reconstruction + DT trigger efficiency plot')
    parser.add_argument('--reco_json_1p2mV', default=os.path.join(_RESULTS_DIR, 'plots', 'efficiency', 'json'))
    parser.add_argument('--reco_json_7p5mV', default=os.path.join(_RESULTS_DIR, 'plots', 'efficiency_7p5mV_ext511', 'json'))
    parser.add_argument('--trig_json',       default=os.path.join(_RESULTS_DIR, 'plots', 'trigger', 'json'))
    parser.add_argument('--output_dir',      default=os.path.join(_RESULTS_DIR, 'plots', 'combined'))
    parser.add_argument('--threshold',       type=float, default=90.0)
    parser.add_argument('--max_energy',      type=float, default=511.0)
    args = parser.parse_args()

    reco_dirs = {'1p2mV': args.reco_json_1p2mV, '7p5mV': args.reco_json_7p5mV}

    for method in ('triexp', 'maxbase'):
        for noise_tag in ('1p2mV', '7p5mV'):
            reco_data = _load_reco(reco_dirs[noise_tag], noise_tag, args.max_energy, method=method)
            trig_data = _load_trigger(args.trig_json, noise_tag, args.max_energy)
            if reco_data is None and trig_data is None:
                print(f'No data found for {noise_tag} ({method})')
                continue
            suffix = '' if method == 'triexp' else f'_{method}'
            out = os.path.join(args.output_dir, f'combined_efficiency{suffix}_{noise_tag}.png')
            plot_combined(noise_tag, reco_data, trig_data, out,
                         threshold_pct=args.threshold, x_max=args.max_energy, method=method)

    print('\nAll done.')


if __name__ == '__main__':
    main()
