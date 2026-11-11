"""QA figures for a denoised test set, organized by energy.

Reads the HDF5 written by ``scripts/denoise_test_set.py`` (denoised + clean +
noisy waveforms and per-window metadata) -- no GPU or model load needed.

Outputs (all under ``--out_dir``):
  examples_single.png   one single-pulse window per typical energy:
                        full window (noisy / clean / denoised), residual
                        (denoised - clean), PSD (noisy / noise / clean /
                        denoised / residual), in mV and mV^2/Hz.
  shape_vs_energy.png   clean waveforms of the single examples normalized to
                        [-1, 1] (divided by max|clean|) and overlaid.
  examples_pileup.png   same layout for pileup windows at several Delta t.
  metrics_vs_energy.png per-window metrics vs energy, binned median and
                        16-84 % band, noisy vs denoised, single vs pileup.
  metrics_vs_snr.png    per-window metrics vs noisy SNR scatter (the layout
                        of ``src/ddpm/inference.py``'s scatter grid, without
                        the 10-shot series).
  metrics_summary.txt   per-energy-bin medians.
  metrics.pkl           per-window metric dicts (cached; delete to recompute).

Metrics are the ones of ``src.ddpm.inference.compute_all_metrics`` plus
baseline RMS (0-1.5 s, before the first onset) and tri-exponential-fit peak
amplitude/time errors relative to the fit of the clean waveform (as in
``src/ddpm/inference.py``). Energy axis: E_1 for single windows; for pileup
window-level metrics E_1 + E_2, and for per-peak metrics each pulse's own
energy (pulse 1 at 1.5 s precedes pulse 2, matching the fit's time order).

Usage:
    python3 -u scripts/plot_triexp_inference.py \\
        --denoised_h5 /media/.../ddpm_triexp_scale_film/denoised_test_000.h5 \\
        --out_dir /media/.../ddpm_triexp_scale_film/qa_test
"""
import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ddpm.inference import compute_all_metrics, _SCATTER_METRICS
from src.basics.fit import fit_pulse

DEFAULT_H5 = ('/media/Disk_YIN/yunshancheng/cuore/ddpm_triexp_scale_film/'
              'denoised_test_000.h5')
TYPICAL_E_KEV = (50, 100, 300, 600, 1000, 2615, 5000)
PILEUP_DT_S = (0.05, 0.3, 1.0, 1.8)
N_BASELINE = 1500  # samples before the first onset (1.5 s at 1 kHz)

_H5 = None


def _open(path):
    global _H5
    if _H5 is None:
        _H5 = h5py.File(path, 'r')
    return _H5


# ── Per-window metrics ──────────────────────────────────────────────────────

def _window_metrics(args):
    path, idx = args
    f = _open(path)
    clean = f['waveforms_clean'][idx].astype(np.float64)
    noisy = f['waveforms_noisy'][idx].astype(np.float64)
    den = f['waveforms'][idx].astype(np.float64)
    scale = float(f['scale'][idx])
    n_peaks = 2 if f['is_pileup'][idx] else 1

    fit_clean = fit_pulse(clean, n_peaks)
    out = {'idx': idx}
    for label, sig in (('noisy', noisy), ('den', den)):
        m = compute_all_metrics(clean, sig, clean / scale, sig / scale)
        m['baseline_rms'] = float(np.sqrt(np.mean(sig[:N_BASELINE] ** 2)))
        m['baseline_rms_norm'] = m['baseline_rms'] / scale
        fr = fit_pulse(sig, n_peaks)
        m['chi2_per_ndf'] = fr.chi2_per_ndf
        m['peaks'] = []
        out[label] = m
        if not (fit_clean.success and fr.success):
            continue
        for i in range(n_peaks):
            c_amp = fit_clean.peak_amps[i]
            m['peaks'].append({
                'err_pct': (fr.peak_amps[i] - c_amp) / c_amp * 100.0
                           if c_amp != 0 else float('nan'),
                'time_err_ms': (fr.peak_times[i]
                                - fit_clean.peak_times[i]) * 1000.0,
            })
    return out


def compute_metrics(path, n_workers):
    with h5py.File(path, 'r') as f:
        n = f['waveforms'].shape[0]
    jobs = [(path, i) for i in range(n)]
    results = []
    with Pool(n_workers) as pool:
        for k, r in enumerate(pool.imap(_window_metrics, jobs, chunksize=16)):
            results.append(r)
            if (k + 1) % 500 == 0:
                print(f'  metrics [{k + 1}/{n}]', flush=True)
    return results


def metric_values(results, src, mdef):
    """Per-window (or per-peak) values of one metric; returns (vals, peak_i).

    peak_i is the pulse index (0/1) for per-peak metrics, else None.
    """
    key = mdef['key']
    tf = mdef.get('transform')
    if mdef.get('is_peak'):
        vals, pidx, widx = [], [], []
        for w, r in enumerate(results):
            for i, pk in enumerate(r[src]['peaks']):
                v = pk['err_pct'] if key == 'peak_amp_err' else pk['time_err_ms']
                vals.append(abs(v))
                pidx.append(i)
                widx.append(w)
        return np.array(vals), np.array(pidx, dtype=int), np.array(widx, dtype=int)
    vals = np.array([r[src][key] for r in results], dtype=np.float64)
    if tf is not None:
        vals = tf(vals)
    return vals, None, np.arange(len(results))


# ── Example figures ─────────────────────────────────────────────────────────

def compute_psd(x, fs, nperseg_cap=2048):
    f, p = welch(x, fs=fs, nperseg=min(nperseg_cap, len(x)),
                 detrend='constant')
    return f[1:], p[1:]


def plot_example_row(axes, clean, noisy, den, fs, title, legend):
    clean, noisy, den = clean * 1e3, noisy * 1e3, den * 1e3
    t = np.arange(clean.size) / fs
    ax_t, ax_r, ax_p = axes

    ax_t.plot(t, noisy, color='tab:red', lw=0.4, alpha=0.6, label='noisy')
    ax_t.plot(t, clean, color='tab:blue', lw=1.2, label='clean')
    ax_t.plot(t, den, color='tab:purple', lw=0.8, ls='--', label='denoised')
    ax_t.set_ylabel('amplitude (mV)')
    ax_t.set_title(title, fontsize=9)

    res = den - clean
    ax_r.plot(t, res, color='tab:purple', lw=0.5)
    ax_r.axhline(0, color='k', lw=0.5)
    ax_r.set_ylabel('denoised - clean (mV)')
    ax_r.set_title(f'residual RMS {np.sqrt(np.mean(res ** 2)):.3f} mV, '
                   f'max |res| {np.max(np.abs(res)):.3f} mV', fontsize=9)

    for x, color, lw, ls, label in (
            (noisy, 'tab:red', 1.2, '-', 'noisy'),
            (noisy - clean, 'tab:gray', 0.8, '-', 'noise'),
            (clean, 'tab:blue', 1.0, '-', 'clean'),
            (den, 'tab:purple', 0.9, '--', 'denoised'),
            (res, 'tab:green', 0.9, '-', 'residual')):
        f, p = compute_psd(x, fs)
        ax_p.loglog(f, p, color=color, lw=lw, ls=ls, label=label, alpha=0.85)
    ax_p.set_ylabel('PSD (mV$^2$/Hz)')
    ax_p.grid(True, which='both', ls=':', lw=0.3, alpha=0.5)

    for ax in (ax_t, ax_r):
        ax.grid(True, alpha=0.3)
    if legend:
        ax_t.legend(fontsize=7, loc='upper right', frameon=False)
        ax_p.legend(fontsize=7, loc='lower left', frameon=False)


def plot_examples(f, idxs, titles, out_png):
    fs = float(f.attrs['f_sample'])
    fig, axes = plt.subplots(len(idxs), 3, figsize=(17, 2.6 * len(idxs)),
                             squeeze=False)
    for row, (idx, title) in enumerate(zip(idxs, titles)):
        plot_example_row(axes[row],
                         f['waveforms_clean'][idx].astype(np.float64),
                         f['waveforms_noisy'][idx].astype(np.float64),
                         f['waveforms'][idx].astype(np.float64),
                         fs, title, legend=(row == 0))
    for ax in axes[-1, :2]:
        ax.set_xlabel('time (s)')
    axes[-1, 2].set_xlabel('frequency (Hz)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')


def plot_shape_comparison(f, idxs, out_png):
    """Clean waveforms of the example windows, each divided by max|clean|
    (so in [-1, 1]), overlaid to compare pulse shape across energy."""
    fs = float(f.attrs['f_sample'])
    E1 = f['energies_1'][:]
    taus = {k: f[f'{k}_1'][:] for k in ('tau_r', 'tau_d1', 'tau_d2')}
    order = sorted(idxs, key=lambda i: E1[i])
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(order)))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for i, c in zip(order, colors):
        clean = f['waveforms_clean'][i].astype(np.float64)
        y = clean / np.max(np.abs(clean))
        t = np.arange(y.size) / fs
        label = (f'{E1[i]:.0f} keV: $\\tau_r$={taus["tau_r"][i] * 1e3:.0f} ms, '
                 f'$\\tau_{{d1}}$={taus["tau_d1"][i] * 1e3:.0f} ms, '
                 f'$\\tau_{{d2}}$={taus["tau_d2"][i]:.2f} s')
        axes[0].plot(t, y, color=c, lw=1.0, label=label)
        axes[1].plot(t, y, color=c, lw=1.0)
        sel = y > 0
        axes[2].semilogy(t[sel], y[sel], color=c, lw=1.0)
    axes[0].set_title('full window')
    axes[1].set_xlim(1.4, 3.5)
    axes[1].set_title('rise and fast decay')
    axes[2].set_ylim(1e-2, 1.5)
    axes[2].set_title('decay, log scale')
    for ax in axes:
        ax.set_xlabel('time (s)')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('clean / max|clean|')
    axes[0].legend(fontsize=7, loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')


def pick_single(E1, is_pu, energies):
    pool = np.where(~is_pu)[0]
    return [int(pool[np.argmin(np.abs(E1[pool] - e))]) for e in energies]


def pick_pileup(E1, E2, dt, is_pu, dts):
    """Closest Delta t among pileups with both energies in 60-1500 keV."""
    pool = np.where(is_pu & (E1 > 60) & (E1 < 1500) & (E2 > 60) & (E2 < 1500))[0]
    if len(pool) < len(dts):
        pool = np.where(is_pu)[0]
    return [int(pool[np.argmin(np.abs(dt[pool] - d))]) for d in dts]


# ── Metric figures ──────────────────────────────────────────────────────────

def binned_quantiles(x, y, edges, min_count=5):
    q = np.full((len(edges) - 1, 3), np.nan)
    b = np.digitize(x, edges) - 1
    for k in range(len(edges) - 1):
        yk = y[(b == k) & np.isfinite(y)]
        if len(yk) >= min_count:
            q[k] = np.percentile(yk, [16, 50, 84])
    return q


def plot_metrics_vs_energy(results, E_win, E_peak, is_pu, edges, out_png):
    centers = np.sqrt(edges[:-1] * edges[1:])
    n = len(_SCATTER_METRICS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)
    series = (('noisy', False, 'single noisy', 'tab:red', '-'),
              ('noisy', True, 'pileup noisy', 'tab:orange', '--'),
              ('den', False, 'single denoised', 'tab:blue', '-'),
              ('den', True, 'pileup denoised', 'tab:purple', '--'))
    for k, mdef in enumerate(_SCATTER_METRICS):
        ax = axes[k // ncols, k % ncols]
        for src, pu, label, color, ls in series:
            vals, pidx, widx = metric_values(results, src, mdef)
            if pidx is None:
                x = E_win
                sel = is_pu == pu
            else:
                x = E_peak[widx, pidx]
                sel = is_pu[widx] == pu
            q = binned_quantiles(x[sel], vals[sel], edges)
            ax.plot(centers, q[:, 1], color=color, ls=ls, marker='o', ms=2.5,
                    lw=1.0, label=label)
            ax.fill_between(centers, q[:, 0], q[:, 2], color=color,
                            alpha=0.12, lw=0)
        ax.set_xscale('log')
        if mdef['key'] not in ('snr', 'psnr'):
            ax.set_yscale('log')
        if mdef.get('ymax') is not None:
            ax.set_ylim(top=mdef['ymax'])
        ax.set_xlabel('energy (keV)', fontsize=8)
        ax.set_ylabel(mdef['label'], fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc='best')
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')
    fig.suptitle('Metrics vs energy (median, 16-84 % band); pileup window '
                 'metrics vs E1+E2, peak metrics vs each pulse energy',
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')


def plot_metrics_vs_snr(results, is_pu, out_png):
    snr_x = np.array([r['noisy']['snr'] for r in results])
    n = len(_SCATTER_METRICS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)
    series = (('noisy', False, 'noisy single', 'tab:red', 'o'),
              ('noisy', True, 'noisy pileup', 'tab:orange', '^'),
              ('den', False, 'denoised single', 'tab:blue', 'o'),
              ('den', True, 'denoised pileup', 'tab:purple', '^'))
    for k, mdef in enumerate(_SCATTER_METRICS):
        ax = axes[k // ncols, k % ncols]
        all_vals = []
        for src, pu, label, color, marker in series:
            vals, _, widx = metric_values(results, src, mdef)
            sel = is_pu[widx] == pu
            ax.scatter(snr_x[widx][sel], vals[sel], s=2, c=color,
                       marker=marker, alpha=0.3, lw=0, label=label)
            all_vals.append(vals[np.isfinite(vals) & (vals > 0)])
        if mdef['key'] not in ('snr', 'psnr'):
            ax.set_yscale('log')
            # Fit-based errors reach ~1e-15 on near-exact fits; clip the
            # axis to the 0.5th percentile so the bulk stays readable.
            ax.set_ylim(bottom=np.percentile(np.concatenate(all_vals), 0.5))
        if mdef.get('ymax') is not None:
            ax.set_ylim(top=mdef['ymax'])
        ax.set_xlabel('noisy SNR (dB)', fontsize=8)
        ax.set_ylabel(mdef['label'], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc='best', markerscale=4)
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')
    fig.suptitle(f'Metrics vs noisy SNR (n_single={int((~is_pu).sum())}, '
                 f'n_pileup={int(is_pu.sum())})', fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')


def write_summary(results, E_win, E_peak, is_pu, edges, out_txt):
    keys = [m for m in _SCATTER_METRICS
            if m['key'] in ('mse', 'mae', 'prd', 'snr', 'psnr', 'cc',
                            'baseline_rms', 'peak_amp_err', 'peak_time_err')]
    lines = []
    for pu, name in ((False, 'single'), (True, 'pileup')):
        lines.append(f'== {name} (median per energy bin; noisy -> denoised) ==')
        head = f'{"E bin (keV)":>16} {"n":>5} ' + ' '.join(
            f'{m["label"]:>22}' for m in keys)
        lines.append(head)
        rows = list(zip(edges[:-1], edges[1:])) + [(edges[0], edges[-1])]
        for lo, hi in rows:
            cells = []
            n_win = int(((E_win >= lo) & (E_win < hi) & (is_pu == pu)).sum())
            for mdef in keys:
                pair = []
                for src in ('noisy', 'den'):
                    vals, pidx, widx = metric_values(results, src, mdef)
                    x = E_win[widx] if pidx is None else E_peak[widx, pidx]
                    sel = (x >= lo) & (x < hi) & (is_pu[widx] == pu)
                    v = vals[sel]
                    v = v[np.isfinite(v)]
                    pair.append(np.median(v) if len(v) else np.nan)
                scl = 1e3 if mdef['key'] in ('mae', 'baseline_rms') else (
                    1e6 if mdef['key'] == 'mse' else 1.0)
                cells.append(f'{pair[0] * scl:10.4g} ->{pair[1] * scl:10.4g}')
            lines.append(f'{lo:7.0f}-{hi:<8.0f} {n_win:5d} ' + ' '.join(
                f'{c:>22}' for c in cells))
        lines.append('')
    lines.append('Units: MSE mV^2, MAE / BL RMS mV, PRD %, SNR/PSNR dB, '
                 '|Amp| err %, |dt| ms. Last row of each block = all energies.')
    with open(out_txt, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'wrote {out_txt}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--denoised_h5', default=DEFAULT_H5)
    p.add_argument('--out_dir', default=None,
                   help='Default: <dir of denoised_h5>/qa_test')
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--n_ebins', type=int, default=12)
    args = p.parse_args()
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.denoised_h5), 'qa_test')
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(args.denoised_h5, 'r') as f:
        E1 = f['energies_1'][:]
        E2 = np.nan_to_num(f['energies_2'][:])
        is_pu = f['is_pileup'][:].astype(bool)
        dt = f['onsets_2'][:] - f['onsets_1'][:]

        idx_s = pick_single(E1, is_pu, TYPICAL_E_KEV)
        plot_examples(f, idx_s,
                      [f'single, E = {E1[i]:.0f} keV (window {i})'
                       for i in idx_s],
                      os.path.join(out_dir, 'examples_single.png'))
        plot_shape_comparison(f, idx_s,
                              os.path.join(out_dir, 'shape_vs_energy.png'))
        idx_p = pick_pileup(E1, E2, dt, is_pu, PILEUP_DT_S)
        plot_examples(f, idx_p,
                      [f'pileup, E1 = {E1[i]:.0f}, E2 = {E2[i]:.0f} keV, '
                       f'$\\Delta t$ = {dt[i] * 1e3:.0f} ms (window {i})'
                       for i in idx_p],
                      os.path.join(out_dir, 'examples_pileup.png'))

    pkl = os.path.join(out_dir, 'metrics.pkl')
    if os.path.exists(pkl):
        with open(pkl, 'rb') as fh:
            results = pickle.load(fh)
        print(f'loaded {pkl}')
    else:
        results = compute_metrics(args.denoised_h5, args.workers)
        with open(pkl, 'wb') as fh:
            pickle.dump(results, fh)
        print(f'wrote {pkl}')

    E_win = np.where(is_pu, E1 + E2, E1)
    E_peak = np.stack([E1, E2], axis=1)
    edges = np.geomspace(30.0, 11000.0, args.n_ebins + 1)
    plot_metrics_vs_energy(results, E_win, E_peak, is_pu, edges,
                           os.path.join(out_dir, 'metrics_vs_energy.png'))
    plot_metrics_vs_snr(results, is_pu,
                        os.path.join(out_dir, 'metrics_vs_snr.png'))
    write_summary(results, E_win, E_peak, is_pu, edges,
                  os.path.join(out_dir, 'metrics_summary.txt'))


if __name__ == '__main__':
    main()
