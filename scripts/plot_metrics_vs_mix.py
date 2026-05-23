"""Scatter: denoising metrics vs noise-mix periodic fraction.

Reads the denoised-test HDF5 produced by ``scripts/denoise_test_set.py``
(which already contains clean, noisy, and 1-shot denoised waveforms plus
the per-window mix metadata from the noise source) and computes the same
metric set used by replot_scatter.py — no GPU needed.

The noise mix uses:
    r      ~ U(0, 1)            — periodic energy fraction
    alpha  = sqrt(r)             — periodic amplitude weight
    beta   = sqrt(1 - r)         — white amplitude weight
so r is the proportion of periodic / structured noise energy.

Usage:
    python3 -u scripts/plot_metrics_vs_mix.py \\
        --denoised_h5 /media/.../ddpm_v3_low/denoised_test_000.h5 \\
        --n 500 --seed 0 \\
        --out_pkl /media/.../ddpm_v3_low/metrics_vs_mix.pkl \\
        --out_png /home/yunshan/cuore/plots/paper/metrics_vs_mix.png
"""
import argparse
import os
import pickle
import sys
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ddpm.inference import compute_all_metrics, _SCATTER_METRICS
from src.basics.fit import fit_pulse


def per_window_metrics(clean_V, noisy_V, den_V, scale, n_peaks):
    """Compute metrics for (noisy, denoised) against clean for one window.

    Fitting policy: tri-exp pulse fits (and the peak/chi2/fit_success
    metrics derived from them) are only performed on single-pulse
    windows. For pileup windows the metric dicts still contain all the
    waveform/spectral metrics (MSE, cosine, SNR, LSD, etc.) but no
    'peaks' / 'chi2_per_ndf' / 'fit_success' entries — downstream
    scatter code skips pileup rows automatically when those keys are
    absent. Reason: the two-pulse tri-exp can assign ~0 amplitude to
    one peak in near-coincident pileups, which sends relative
    amplitude error toward infinity.
    """
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(np.max(np.abs(noisy_V))), 1e-12)
    x_clean_norm = clean_V / scale
    x_noisy_norm = noisy_V / scale
    x_den_norm = den_V / scale

    m_noisy = compute_all_metrics(clean_V, noisy_V, x_clean_norm, x_noisy_norm)
    m_single = compute_all_metrics(clean_V, den_V, x_clean_norm, x_den_norm)

    if n_peaks == 1:
        fit_clean = fit_pulse(clean_V, 1)
        for label, signal_p, m in [('noisy', noisy_V, m_noisy),
                                    ('single', den_V, m_single)]:
            fr = fit_pulse(signal_p, 1)
            m['peaks'] = []
            m['chi2_per_ndf'] = fr.chi2_per_ndf
            m['fit_success'] = fr.success
            c_amp = fit_clean.peak_amps[0]
            s_amp = fr.peak_amps[0]
            c_pos_s = fit_clean.peak_times[0]
            s_pos_s = fr.peak_times[0]
            err_pct = (s_amp - c_amp) / c_amp * 100.0 if c_amp != 0 else 0.0
            time_err_ms = (s_pos_s - c_pos_s) * 1000.0
            m['peaks'].append({
                'clean_amp': c_amp,
                'signal_amp': s_amp,
                'err_pct': err_pct,
                'clean_pos': c_pos_s * 1000.0,
                'signal_pos': s_pos_s * 1000.0,
                'time_err_ms': time_err_ms,
            })
    # else: pileup → no fit, no 'peaks'/'chi2'/'fit_success' fields.

    n_baseline = 1500
    for signal_p, signal_n, m in [
            (noisy_V, x_noisy_norm, m_noisy),
            (den_V,   x_den_norm,   m_single)]:
        m['baseline_rms'] = float(np.sqrt(np.mean(signal_p[:n_baseline] ** 2)))
        m['baseline_rms_norm'] = float(np.sqrt(np.mean(signal_n[:n_baseline] ** 2)))

    return m_noisy, m_single


def collect_xy(results, mkey, metric_key, is_peak, transform=None):
    """x = mix r (periodic energy fraction), y = metric.

    For fit-derived metrics (peak_*, chi2_per_ndf, fit_success), pileup
    windows have no fit results — skipped automatically.
    """
    xs, ys = [], []
    for r in results:
        x_window = r['mix_r']
        m = r[mkey]
        if is_peak:
            for pk in m.get('peaks', []):
                if metric_key == 'peak_amp_err':
                    v = pk['err_pct']
                elif metric_key == 'peak_time_err':
                    v = pk['time_err_ms']
                else:
                    continue
                xs.append(x_window)
                ys.append(v)
        else:
            if metric_key not in m:
                continue
            v = m[metric_key]
            if np.isfinite(v):
                xs.append(x_window)
                ys.append(v)
    ys = np.array(ys)
    if transform is not None and len(ys):
        ys = transform(ys)
    return np.array(xs), ys


def plot_grid(results, out_path, title_suffix=''):
    """Only the denoised series is plotted (purple). Noisy is the baseline
    everyone already knows — this figure is about how denoising quality
    moves as the periodic/white mix changes."""
    sources = [
        ('m_single', '1-shot', 'tab:purple'),
    ]
    n = len(_SCATTER_METRICS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)

    for k, mdef in enumerate(_SCATTER_METRICS):
        ax = axes[k // ncols, k % ncols]
        is_peak = mdef.get('is_peak', False)
        tf = mdef.get('transform', None)
        finite_p95 = []
        for mkey, name, color in sources:
            xs, ys = collect_xy(results, mkey, mdef['key'], is_peak, transform=tf)
            ax.scatter(xs, ys, s=6, c=color, alpha=0.5, label=name)
            if len(ys):
                ys_f = ys[np.isfinite(ys)]
                if len(ys_f):
                    finite_p95.append(np.percentile(ys_f, 95))
        ax.set_xlabel('periodic energy fraction r = $\\alpha^2$', fontsize=8)
        ax.set_ylabel(mdef['label'], fontsize=8)
        if finite_p95:
            ax.set_ylim(top=1.2 * max(finite_p95))
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc='best')
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')

    fig.suptitle(f'Metrics vs noise-mix periodic fraction{title_suffix}', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    # PNG sibling for previews if user passes a .pdf
    if out_path.endswith('.pdf'):
        plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--denoised_h5', required=True,
                   help='denoised_test_000.h5 from denoise_test_set.py')
    p.add_argument('--n', type=int, default=500,
                   help='Number of windows to evaluate')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_pkl', required=True)
    p.add_argument('--out_png', required=True)
    p.add_argument('--title_suffix', default='')
    args = p.parse_args()

    with h5py.File(args.denoised_h5, 'r') as f:
        n_all = int(f.attrs['n_windows'])
        is_pileup = f['is_pileup'][:]
        mix_r = f['noise/mix/r'][:]
        mix_alpha = f['noise/mix/alpha'][:]
        mix_beta = f['noise/mix/beta'][:]
        mix_scale = f['noise/mix/scale'][:]
        scale_arr = f['scale'][:]

        rng = np.random.default_rng(args.seed)
        n_eval = min(args.n, n_all)
        indices = rng.choice(n_all, size=n_eval, replace=False)
        indices = np.sort(indices)
        print(f'Evaluating {n_eval}/{n_all} windows.')

        results = []
        t0 = time.time()
        wfs_clean = f['waveforms_clean']
        wfs_noisy = f['waveforms_noisy']
        wfs_den = f['waveforms']
        for k, idx in enumerate(indices):
            idx = int(idx)
            clean_V = wfs_clean[idx].astype(np.float64)
            noisy_V = wfs_noisy[idx].astype(np.float64)
            den_V = wfs_den[idx].astype(np.float64)
            pileup = bool(is_pileup[idx])
            n_peaks = 2 if pileup else 1
            try:
                m_noisy, m_single = per_window_metrics(
                    clean_V, noisy_V, den_V, float(scale_arr[idx]), n_peaks)
            except Exception as e:
                print(f'  [skip {idx}] {e}')
                continue
            m_noisy['is_pileup'] = pileup
            m_single['is_pileup'] = pileup
            results.append({
                'idx': idx,
                'mix_r': float(mix_r[idx]),
                'mix_alpha': float(mix_alpha[idx]),
                'mix_beta': float(mix_beta[idx]),
                'mix_scale': float(mix_scale[idx]),
                'is_pileup': pileup,
                'm_noisy': m_noisy,
                'm_single': m_single,
            })
            if (k + 1) % 50 == 0 or k == 0:
                elapsed = time.time() - t0
                rate = (k + 1) / elapsed
                eta = (n_eval - k - 1) / rate if rate > 0 else 0
                print(f'  [{k+1}/{n_eval}] r={results[-1]["mix_r"]:.2f} '
                      f'(rate={rate:.2f}/s, ETA={eta:.0f}s)')

    payload = {
        'meta': {
            'denoised_h5': args.denoised_h5,
            'n_requested': args.n,
            'n_evaluated': len(results),
            'seed': args.seed,
            'title_suffix': args.title_suffix,
        },
        'results': results,
    }
    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(payload, f)
    print(f'Saved pickle: {args.out_pkl} ({len(results)} windows)')

    plot_grid(results, args.out_png, title_suffix=args.title_suffix)


if __name__ == '__main__':
    main()
