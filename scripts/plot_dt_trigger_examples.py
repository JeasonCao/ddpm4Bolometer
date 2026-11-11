"""
Illustrative examples of the BLIND DT (derivative-threshold) trigger
decision.

For a handful of random events at a given energy/noise level, shows the
waveform (zoomed near the pulse onset) and its derivative, with the
ROLLING (causal, never-looks-ahead) threshold and any trigger(s) found —
noisy vs denoised side by side. The true onset is drawn only as a reference
line for scoring; the trigger decision itself never uses it. Triggers that
fall inside the scoring/match window are shown green ("matched" — this
pulse was caught); any others are orange ("extra" — unrelated to this pulse,
e.g. dark-rate or model artifacts, not used in the efficiency score).

Usage
-----
python scripts/plot_dt_trigger_examples.py \
    --eval_file /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p2mV/eval_efficiency_5keV.h5 \
    --n_examples 4 --seed 0 \
    --output results/plots/trigger/examples_5keV_1p2mV.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.basics.trigger import dt_trigger_blind
from src.basics.plot_utils import timestamped_path

ROLLING_WINDOW_MS = 300.0
DEAD_TIME_MS      = 300.0
MATCH_PRE_MS  = 100.0
MATCH_POST_MS = 200.0
ZOOM_PRE_MS  = 60.0
ZOOM_POST_MS = 350.0


def _plot_pair(fig, gs_row, t, x, dv, threshold, fs, onset_idx, window_n,
              trigger_idxs, match_pre_n, match_post_n, zoom_lo, zoom_hi, title, color):
    ax_wave = fig.add_subplot(gs_row[0])
    ax_dv   = fig.add_subplot(gs_row[1], sharex=ax_wave)

    onset_ms = t[onset_idx] * 1e3
    rel_ms = t * 1e3 - onset_ms

    ax_wave.plot(rel_ms[zoom_lo:zoom_hi], x[zoom_lo:zoom_hi] * 1e3, color=color, lw=1.0)
    ax_wave.axvline(0, color='gray', ls=':', lw=1.0, label='true onset (scoring only)')
    ax_wave.set_ylabel('Amp [mV]', fontsize=8)
    ax_wave.set_title(title, fontsize=9)
    ax_wave.tick_params(labelsize=7)
    ax_wave.grid(alpha=0.3)

    # dv[k] <-> original index k+1; threshold[k] <-> dv[window_n+k] <-> original index window_n+k+1
    dv_rel_ms = rel_ms[1:]
    ax_dv.plot(dv_rel_ms[zoom_lo:zoom_hi - 1], dv[zoom_lo:zoom_hi - 1] * 1e3, color=color, lw=1.0)

    thr_lo = max(zoom_lo - window_n - 1, 0)
    thr_hi = max(zoom_hi - 1 - window_n, 0)
    if thr_hi > thr_lo:
        thr_orig_idx = np.arange(thr_lo, thr_hi) + window_n + 1
        thr_rel_ms = thr_orig_idx / fs * 1e3 - onset_ms
        ax_dv.plot(thr_rel_ms, threshold[thr_lo:thr_hi] * 1e3, color='black', ls='--',
                  lw=0.9, label='rolling threshold')

    ax_dv.axvline(0, color='gray', ls=':', lw=1.0)
    ax_dv.set_ylabel('dV [mV/sample]', fontsize=8)
    ax_dv.set_xlabel('Time rel. to true onset [ms]', fontsize=8)
    ax_dv.tick_params(labelsize=7)
    ax_dv.grid(alpha=0.3)

    matched_idx = None
    n_extra = 0
    for ti in trigger_idxs:
        tt_ms = ti / fs * 1e3 - onset_ms
        is_match = (ti >= onset_idx - match_pre_n) and (ti <= onset_idx + match_post_n)
        if is_match and matched_idx is None:
            matched_idx = ti
        if not is_match:
            n_extra += 1
        lcolor = 'limegreen' if is_match else 'darkorange'
        if zoom_lo <= ti < zoom_hi:
            ax_wave.axvline(tt_ms, color=lcolor, lw=1.6)
            ax_dv.axvline(tt_ms, color=lcolor, lw=1.6)

    if matched_idx is not None:
        status = f'MATCHED @ {matched_idx/fs*1e3 - onset_ms:+.1f} ms'
        if n_extra:
            status += f'  (+{n_extra} extra)'
        status_color = 'darkgreen'
    elif n_extra:
        status = f'NOT MATCHED  ({n_extra} extra trigger(s) elsewhere)'
        status_color = 'darkorange'
    else:
        status = 'NOT TRIGGERED'
        status_color = 'firebrick'

    ax_wave.text(0.98, 0.05, status, transform=ax_wave.transAxes,
                ha='right', va='bottom', fontsize=8, color=status_color, fontweight='bold')
    ax_wave.legend(fontsize=6.5, loc='upper left')


def run(eval_file, n_examples, seed, output, nsigma, n_above,
       window_ms, dead_ms, match_pre_ms, match_post_ms):
    with h5py.File(eval_file, 'r') as f:
        energy_kev = float(f.attrs['energy_kev'])
        fs         = float(f.attrs['f_sample'])
        onset_s    = float(f.attrs['onset_s'])
        noisy_all    = f['waveforms_noisy']
        denoised_all = f['waveforms_denoised']
        n_total = noisy_all.shape[0]

        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=n_examples, replace=False)
        idx.sort()

        noisy    = noisy_all[idx.tolist()]
        denoised = denoised_all[idx.tolist()]

    onset_idx = int(round(onset_s * fs))
    window_n  = int(round(window_ms * 1e-3 * fs))
    dead_n    = int(round(dead_ms * 1e-3 * fs))
    match_pre_n  = int(round(match_pre_ms * 1e-3 * fs))
    match_post_n = int(round(match_post_ms * 1e-3 * fs))

    zoom_lo = max(0, onset_idx - int(ZOOM_PRE_MS * 1e-3 * fs))
    zoom_hi = min(noisy.shape[1], onset_idx + int(ZOOM_POST_MS * 1e-3 * fs))

    n_rows = n_examples * 2
    fig = plt.figure(figsize=(11, 2.6 * n_examples))
    gs = gridspec.GridSpec(n_rows, 2, height_ratios=[2, 1.3] * n_examples,
                           hspace=0.6, wspace=0.3)

    t = np.arange(noisy.shape[1]) / fs

    kw = dict(nsigma=nsigma, n_above=n_above, dead_samples=dead_n)
    for i in range(n_examples):
        idxs_n, _, _, thr_n = dt_trigger_blind(noisy[i],    fs, window_n, **kw)
        idxs_d, _, _, thr_d = dt_trigger_blind(denoised[i], fs, window_n, **kw)
        dv_n = np.diff(noisy[i])
        dv_d = np.diff(denoised[i])

        gs_row_n = [gs[2 * i, 0], gs[2 * i + 1, 0]]
        gs_row_d = [gs[2 * i, 1], gs[2 * i + 1, 1]]

        _plot_pair(fig, gs_row_n, t, noisy[i], dv_n, thr_n, fs, onset_idx, window_n,
                  idxs_n, match_pre_n, match_post_n, zoom_lo, zoom_hi,
                  f'Event #{idx[i]} — Noisy', 'steelblue')
        _plot_pair(fig, gs_row_d, t, denoised[i], dv_d, thr_d, fs, onset_idx, window_n,
                  idxs_d, match_pre_n, match_post_n, zoom_lo, zoom_hi,
                  f'Event #{idx[i]} — Denoised', 'tomato')

    fig.suptitle(
        f'Blind DT Trigger Examples — {energy_kev:.0f} keV  '
        f'(nsigma={nsigma}, n_above={n_above}, rolling window={window_ms:.0f} ms)',
        fontsize=13, y=1.0)

    output = timestamped_path(output)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {output}')


def main():
    parser = argparse.ArgumentParser(description='Illustrative blind DT trigger examples')
    parser.add_argument('--eval_file', required=True,
                        help='Eval HDF5 with waveforms_noisy + waveforms_denoised')
    parser.add_argument('--n_examples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default=None)
    parser.add_argument('--nsigma',  type=float, default=4.0)
    parser.add_argument('--n_above', type=int,   default=5)
    parser.add_argument('--window_ms', type=float, default=ROLLING_WINDOW_MS)
    parser.add_argument('--dead_ms',   type=float, default=DEAD_TIME_MS)
    parser.add_argument('--match_pre_ms',  type=float, default=MATCH_PRE_MS)
    parser.add_argument('--match_post_ms', type=float, default=MATCH_POST_MS)
    args = parser.parse_args()

    output = args.output or os.path.join(
        _RESULTS_DIR, 'plots', 'trigger',
        f'examples_{os.path.splitext(os.path.basename(args.eval_file))[0]}.png')

    run(args.eval_file, args.n_examples, args.seed, output, args.nsigma, args.n_above,
       args.window_ms, args.dead_ms, args.match_pre_ms, args.match_post_ms)


if __name__ == '__main__':
    main()
