"""
Plot reconstruction efficiency vs energy with polynomial fits and threshold line.

Reads the JSON summary files produced by plot_efficiency.py and generates
4 figures: 2 noise levels × 2 reconstruction methods (tri-exp / max-baseline).

Usage
-----
python scripts/plot_efficiency_threshold.py \
    --json_dir    results/plots/efficiency/json \
    --output_dir  results/plots/efficiency \
    --threshold   95 \
    --max_energy  40
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

# ── noise label helpers ───────────────────────────────────────────────────────

_NOISE_LABELS = {
    '1p2mV': '1.2 mV',
    '7p5mV': '7.5 mV',
}

def _noise_rms_str(tag: str) -> str:
    return _NOISE_LABELS.get(tag, tag)


# ── polynomial fit + threshold crossing ───────────────────────────────────────

def _trailing_saturated_run(efficiencies, sat_threshold=99.5):
    """Length of the trailing run of points >= sat_threshold (assumes energies sorted ascending)."""
    run = 0
    for v in reversed(efficiencies):
        if v >= sat_threshold:
            run += 1
        else:
            break
    return run


def _fit_and_threshold(energies, efficiencies, deg=4, threshold_pct=95.0,
                       sat_threshold=99.5):
    """
    Fit a degree-`deg` polynomial in log(E) space.

    A long trailing run of saturated (~100%) points distorts a high-degree
    polynomial (oscillation). If >=2 trailing points are saturated, drop all
    but one anchor point from the fit — the curve is only drawn up to that
    anchor's energy; the (still 100%-clustered) points beyond it are left to
    the scatter markers only.

    Returns (poly1d in log space, log_e_min, log_e_max, threshold_energy or None).
    """
    run = _trailing_saturated_run(efficiencies, sat_threshold)
    n = len(efficiencies)
    cutoff = n - run + 1 if run >= 2 else n
    cutoff = max(cutoff, min(deg + 2, n))   # keep enough points to fit `deg`

    log_e_fit = np.log(energies[:cutoff])
    eff_fit   = efficiencies[:cutoff]
    use_deg   = min(deg, len(eff_fit) - 1)

    coeffs = np.polyfit(log_e_fit, eff_fit, use_deg)
    p = np.poly1d(coeffs)

    log_e_min, log_e_max = log_e_fit.min(), log_e_fit.max()
    e_dense = np.linspace(log_e_min, log_e_max, 2000)
    vals = p(e_dense)

    # Find the lowest energy where efficiency crosses threshold_pct from below
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

def plot_threshold(energies, eff_before, eff_after,
                   noise_tag, method, energy_cut,
                   output_path, poly_deg=4, threshold_pct=95.0):

    noise_str = _noise_rms_str(noise_tag)
    method_label = 'Tri-exp fit' if method == 'triexp' else 'Max-baseline'
    cut_pct = int(round(energy_cut * 100))

    # Each series gets its own fit range — a long trailing 100% run is
    # excluded from the fit (see _fit_and_threshold), so before/after curves
    # may end at different energies.
    p_before, leb_min, leb_max, thr_before = _fit_and_threshold(energies, eff_before, poly_deg, threshold_pct)
    p_after,  lea_min, lea_max, thr_after  = _fit_and_threshold(energies, eff_after,  poly_deg, threshold_pct)

    log_e_plot_before = np.linspace(leb_min, leb_max, 500)
    log_e_plot_after  = np.linspace(lea_min, lea_max, 500)
    e_plot_before = np.exp(log_e_plot_before)
    e_plot_after  = np.exp(log_e_plot_after)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter — "before" drawn larger/behind so it remains visible as a ring
    # even where before/after sit at the same (energy, 100%) point.
    ax.scatter(energies, eff_before, color='steelblue', zorder=4,
               s=55, alpha=0.85, edgecolors='white', linewidth=0.6,
               label='Before denoising')
    ax.scatter(energies, eff_after,  color='tomato',    zorder=5,
               s=25, edgecolors='white', linewidth=0.4,
               label='After denoising')

    # Polynomial curves — semi-transparent red, dashed=before / solid=after
    ax.plot(e_plot_before, p_before(log_e_plot_before), color='crimson', alpha=0.45, lw=2.0,
            ls='--', label=f'Poly-{poly_deg} fit (before)')
    ax.plot(e_plot_after,  p_after(log_e_plot_after),  color='crimson', alpha=0.45, lw=2.0,
            ls='-',  label=f'Poly-{poly_deg} fit (after)')

    # Threshold line
    thr_label = f'{threshold_pct:.0f}% threshold'
    ax.axhline(threshold_pct, color='black', ls='--', lw=1.2, label=thr_label)

    # Threshold intersections — vertical dashed line only (no dot); energy label at x-axis
    for thr, color, who in [(thr_before, 'steelblue', 'Before'),
                             (thr_after,  'tomato',    'After')]:
        if thr is not None:
            ax.axvline(thr, color=color, ls=':', lw=1.5)
            # Annotate energy value just below the x-axis, in data-x / axes-y coords
            ax.annotate(f'{thr:.1f} keV\n({who})',
                        xy=(thr, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -6), textcoords='offset points',
                        ha='center', va='top', fontsize=8, color=color,
                        annotation_clip=False)

    # Log x-axis — avoids tick crowding when the energy range spans 1–500+ keV
    ax.set_xscale('log')
    ax.set_xlim(energies.min() * 0.7, energies.max() * 1.4)
    ax.tick_params(axis='x', which='major', labelsize=9)

    ax.set_ylim(-5, 110)
    ax.set_xlabel('Energy (keV)', fontsize=11)
    ax.set_ylabel('Efficiency (%)', fontsize=11)
    ax.set_title(
        f'Reconstruction Efficiency vs Energy\n'
        f'{method_label}  |  noise RMS = {noise_str}  |  |ΔE/E| < {cut_pct}%',
        fontsize=11
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {output_path}')
    if thr_before is not None:
        print(f'  Before threshold: {thr_before:.2f} keV')
    if thr_after is not None:
        print(f'  After  threshold: {thr_after:.2f} keV')


# ── batch runner ──────────────────────────────────────────────────────────────

def run(json_dir, output_dir, max_energy=None, threshold_pct=95.0):
    json_files = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    if not json_files:
        print(f'No JSON files found in {json_dir}')
        return

    # Group by noise tag (1p2mV / 7p5mV).
    # Skip "double-suffix" new-format files (e.g. *_1p2mV_1p2mV_*) — they use
    # global calibration over all 17 energy points, which biases low-energy efficiency.
    groups = {}
    for jf in json_files:
        bn = os.path.basename(jf)
        # filename: efficiency_5keV_1p2mV_t7pct_m7pct.json
        for noise_tag in _NOISE_LABELS:
            if f'_{noise_tag}_' in bn:
                # Skip new double-suffix format
                if bn.count(noise_tag) > 1:
                    break
                groups.setdefault(noise_tag, []).append(jf)
                break

    for noise_tag, files in sorted(groups.items()):
        records = []
        for jf in files:
            with open(jf) as f:
                records.append(json.load(f))
        records.sort(key=lambda r: r['energy_kev'])

        # Filter to low-energy range if requested
        if max_energy is not None:
            records = [r for r in records if r['energy_kev'] <= max_energy]
        if not records:
            print(f'No records after max_energy filter for {noise_tag}')
            continue

        energies     = np.array([r['energy_kev']          for r in records])
        eff_tri_b    = np.array([r['eff_triexp_before']    for r in records])
        eff_tri_a    = np.array([r['eff_triexp_after']     for r in records])
        eff_mb_b     = np.array([r['eff_maxbase_before']   for r in records])
        eff_mb_a     = np.array([r['eff_maxbase_after']    for r in records])
        cut_triexp   = records[0]['energy_cut_triexp']
        cut_maxbase  = records[0]['energy_cut_maxbase']

        thr_suffix = f'_thr{int(threshold_pct)}'
        max_e_suffix = f'_max{int(max_energy)}keV' if max_energy is not None else ''

        for method, eff_b, eff_a, cut in [
            ('triexp',   eff_tri_b, eff_tri_a, cut_triexp),
            ('maxbase',  eff_mb_b,  eff_mb_a,  cut_maxbase),
        ]:
            out = os.path.join(output_dir,
                               f'threshold_{method}_{noise_tag}{thr_suffix}{max_e_suffix}.png')
            plot_threshold(energies, eff_b, eff_a,
                           noise_tag, method, cut, out,
                           threshold_pct=threshold_pct)


def main():
    parser = argparse.ArgumentParser(
        description='Efficiency vs energy + threshold curves')
    parser.add_argument('--json_dir',    default=None)
    parser.add_argument('--output_dir',  default=None)
    parser.add_argument('--threshold',   type=float, default=95.0,
                        help='Efficiency threshold in percent (default: 95)')
    parser.add_argument('--max_energy',  type=float, default=None,
                        help='Only use JSON files with energy_kev <= this value')
    args = parser.parse_args()

    json_dir   = args.json_dir   or os.path.join(_RESULTS_DIR, 'plots', 'efficiency', 'json')
    output_dir = args.output_dir or os.path.join(_RESULTS_DIR, 'plots', 'efficiency')

    run(json_dir, output_dir,
        max_energy=args.max_energy,
        threshold_pct=args.threshold)
    print('\nAll done.')


if __name__ == '__main__':
    main()
