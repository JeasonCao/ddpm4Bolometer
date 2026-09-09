"""
Plot DT trigger efficiency vs energy, before vs after DDPM denoising.

Reads the JSON summaries produced by run_dt_trigger.py and generates one
figure per noise level (log-x energy axis, polynomial fit + threshold
crossing — same conventions as plot_efficiency_threshold.py).

Usage
-----
python scripts/plot_trigger_efficiency.py \
    --json_dir   results/plots/trigger/json \
    --output_dir results/plots/trigger \
    --threshold  95
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
from scipy.optimize import brentq

from src.basics.plot_utils import timestamped_path

_NOISE_LABELS = {'1p2mV': '1.2 mV', '7p5mV': '7.5 mV'}


# ── polynomial fit + threshold crossing (saturated-tail truncated) ───────────

def _trailing_saturated_run(efficiencies, sat_threshold=99.5):
    run = 0
    for v in reversed(efficiencies):
        if v >= sat_threshold:
            run += 1
        else:
            break
    return run


def _fit_and_threshold(energies, efficiencies, deg=4, threshold_pct=95.0,
                       sat_threshold=99.5):
    """Fit a degree-`deg` polynomial in log(E) space, dropping all but one
    anchor point of a long trailing ~100% run (avoids high-degree oscillation).
    Returns (poly1d, log_e_min, log_e_max, threshold_energy or None)."""
    run = _trailing_saturated_run(efficiencies, sat_threshold)
    n = len(efficiencies)
    cutoff = n - run + 1 if run >= 2 else n
    cutoff = max(cutoff, min(deg + 2, n))

    log_e_fit = np.log(energies[:cutoff])
    eff_fit   = efficiencies[:cutoff]
    use_deg   = min(deg, len(eff_fit) - 1)

    coeffs = np.polyfit(log_e_fit, eff_fit, use_deg)
    p = np.poly1d(coeffs)

    log_e_min, log_e_max = log_e_fit.min(), log_e_fit.max()
    e_dense = np.linspace(log_e_min, log_e_max, 2000)
    vals = p(e_dense)

    threshold = None
    for i in range(len(vals) - 1):
        if (vals[i] - threshold_pct) * (vals[i + 1] - threshold_pct) < 0 and vals[i] < threshold_pct:
            try:
                log_thr = brentq(lambda x: p(x) - threshold_pct, e_dense[i], e_dense[i + 1])
                threshold = np.exp(log_thr)
            except ValueError:
                pass
            break

    return p, log_e_min, log_e_max, threshold


# ── single figure ─────────────────────────────────────────────────────────────

def plot_trigger(energies, eff_before, eff_after, noise_tag, output_path,
                 poly_deg=4, threshold_pct=95.0):
    noise_str = _NOISE_LABELS.get(noise_tag, noise_tag)

    p_before, leb_min, leb_max, thr_before = _fit_and_threshold(energies, eff_before, poly_deg, threshold_pct)
    p_after,  lea_min, lea_max, thr_after  = _fit_and_threshold(energies, eff_after,  poly_deg, threshold_pct)

    log_e_plot_before = np.linspace(leb_min, leb_max, 500)
    log_e_plot_after  = np.linspace(lea_min, lea_max, 500)
    e_plot_before = np.exp(log_e_plot_before)
    e_plot_after  = np.exp(log_e_plot_after)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(energies, eff_before, color='steelblue', zorder=4,
              s=55, alpha=0.85, edgecolors='white', linewidth=0.6,
              label='Before denoising')
    ax.scatter(energies, eff_after,  color='tomato',    zorder=5,
              s=25, edgecolors='white', linewidth=0.4,
              label='After denoising')

    ax.plot(e_plot_before, p_before(log_e_plot_before), color='crimson', alpha=0.45, lw=2.0,
           ls='--', label=f'Poly-{poly_deg} fit (before)')
    ax.plot(e_plot_after,  p_after(log_e_plot_after),  color='crimson', alpha=0.45, lw=2.0,
           ls='-',  label=f'Poly-{poly_deg} fit (after)')

    thr_label = f'{threshold_pct:.0f}% threshold'
    ax.axhline(threshold_pct, color='black', ls='--', lw=1.2, label=thr_label)

    for thr, color, who in [(thr_before, 'steelblue', 'Before'),
                            (thr_after,  'tomato',    'After')]:
        if thr is not None:
            ax.axvline(thr, color=color, ls=':', lw=1.5)
            ax.annotate(f'{thr:.1f} keV\n({who})',
                       xy=(thr, 0), xycoords=('data', 'axes fraction'),
                       xytext=(0, -6), textcoords='offset points',
                       ha='center', va='top', fontsize=8, color=color,
                       annotation_clip=False)

    ax.set_xscale('log')
    ax.set_xlim(energies.min() * 0.7, energies.max() * 1.4)
    ax.set_ylim(-5, 110)
    ax.set_xlabel('Energy (keV)', fontsize=11)
    ax.set_ylabel('Trigger efficiency (%)', fontsize=11)
    ax.set_title(
        f'DT Trigger Efficiency vs Energy\n'
        f'noise RMS = {noise_str}  |  DT trigger: nsigma=4, n_above=5 samples',
        fontsize=11
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    output_path = timestamped_path(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {output_path}')
    if thr_before is not None:
        print(f'  Before threshold: {thr_before:.2f} keV')
    if thr_after is not None:
        print(f'  After  threshold: {thr_after:.2f} keV')


# ── batch runner ──────────────────────────────────────────────────────────────

def run(json_dir, output_dir, threshold_pct=95.0):
    json_files = sorted(glob.glob(os.path.join(json_dir, 'trigger_*.json')))
    if not json_files:
        print(f'No trigger JSON files found in {json_dir}')
        return

    groups = {}
    for jf in json_files:
        bn = os.path.basename(jf)
        for noise_tag in _NOISE_LABELS:
            if f'_{noise_tag}' in bn:
                groups.setdefault(noise_tag, []).append(jf)
                break

    for noise_tag, files in sorted(groups.items()):
        records = []
        for jf in files:
            with open(jf) as f:
                records.append(json.load(f))
        records.sort(key=lambda r: r['energy_kev'])

        energies   = np.array([r['energy_kev']           for r in records])
        eff_before = np.array([r['trigger_eff_before']    for r in records])
        eff_after  = np.array([r['trigger_eff_after']     for r in records])

        out = os.path.join(output_dir, f'trigger_efficiency_{noise_tag}.png')
        plot_trigger(energies, eff_before, eff_after, noise_tag, out,
                    threshold_pct=threshold_pct)


def main():
    parser = argparse.ArgumentParser(
        description='DT trigger efficiency vs energy, before vs after denoising')
    parser.add_argument('--json_dir',   default=os.path.join(_RESULTS_DIR, 'plots', 'trigger', 'json'))
    parser.add_argument('--output_dir', default=os.path.join(_RESULTS_DIR, 'plots', 'trigger'))
    parser.add_argument('--threshold',  type=float, default=95.0)
    args = parser.parse_args()

    run(args.json_dir, args.output_dir, args.threshold)
    print('\nAll done.')


if __name__ == '__main__':
    main()
