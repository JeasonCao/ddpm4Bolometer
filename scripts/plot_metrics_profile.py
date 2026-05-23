"""Profile (binned mean ± SEM) version of the metrics-vs-mix scatter.

Reads the pickle that ``scripts/plot_metrics_vs_mix.py`` already wrote —
no re-inference or re-fit. For each metric, bins r ∈ [0, 1] into N
equal-width bins and plots one error-bar point per bin (mean ± standard
error of the mean over windows in that bin).

Usage
-----
    python3 -u scripts/plot_metrics_profile.py \\
        --results_pkl /media/.../ddpm_v3_low/metrics_vs_mix.pkl \\
        --out_png /home/yunshan/cuore/plots/paper/metrics_vs_mix_profile.png \\
        --n_bins 10
"""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ddpm.inference import _SCATTER_METRICS
from scripts.plot_metrics_vs_mix import collect_xy


def profile_bins(xs, ys, n_bins=10, x_lo=0.0, x_hi=1.0, min_count=5):
    """Equal-width binning. Returns (centers, means, sem, counts);
    bins with < min_count points get NaN entries."""
    edges = np.linspace(x_lo, x_hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    sem = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    if len(xs) == 0:
        return centers, means, sem, counts
    bin_idx = np.clip(np.digitize(xs, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        sel = (bin_idx == b)
        ys_b = ys[sel]
        ys_b = ys_b[np.isfinite(ys_b)]
        counts[b] = len(ys_b)
        if counts[b] >= min_count:
            means[b] = float(np.mean(ys_b))
            sem[b] = (float(np.std(ys_b, ddof=1) / np.sqrt(counts[b]))
                      if counts[b] > 1 else 0.0)
    return centers, means, sem, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_pkl', required=True)
    p.add_argument('--out_png', required=True)
    p.add_argument('--n_bins', type=int, default=10)
    p.add_argument('--min_count', type=int, default=5,
                   help='Bins with fewer windows than this are skipped')
    p.add_argument('--title_suffix', default='')
    args = p.parse_args()

    with open(args.results_pkl, 'rb') as f:
        payload = pickle.load(f)
    results = payload['results']
    print(f'Loaded {len(results)} windows from {args.results_pkl}')

    sources = [('m_single', '1-shot', 'tab:purple')]

    n = len(_SCATTER_METRICS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)

    for k, mdef in enumerate(_SCATTER_METRICS):
        ax = axes[k // ncols, k % ncols]
        is_peak = mdef.get('is_peak', False)
        tf = mdef.get('transform', None)
        finite_tops = []
        finite_bots = []
        for mkey, name, color in sources:
            xs, ys = collect_xy(results, mkey, mdef['key'], is_peak, transform=tf)
            centers, means, sem, counts = profile_bins(
                xs, ys, n_bins=args.n_bins, x_lo=0.0, x_hi=1.0,
                min_count=args.min_count)
            valid = ~np.isnan(means)
            ax.errorbar(centers[valid], means[valid], yerr=sem[valid],
                        fmt='o-', color=color, ecolor=color,
                        elinewidth=1.0, capsize=2, markersize=4,
                        label=f'{name} (n_pts={int(counts[valid].sum())})')
            if valid.any():
                finite_tops.append(float(np.max(means[valid] + sem[valid])))
                finite_bots.append(float(np.min(means[valid] - sem[valid])))
        ax.set_xlabel('periodic energy fraction r = $\\alpha^2$', fontsize=8)
        ax.set_ylabel(mdef['label'], fontsize=8)
        if finite_tops:
            top = max(finite_tops)
            bot = min(finite_bots)
            pad = 0.1 * max(abs(top), abs(bot), 1e-12)
            ax.set_ylim(top=top + pad, bottom=bot - pad)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc='best')
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')

    fig.suptitle(
        f'Metrics vs noise-mix periodic fraction — profile '
        f'(mean±SEM, {args.n_bins} bins){args.title_suffix}',
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.savefig(args.out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {args.out_png}')


if __name__ == '__main__':
    main()
