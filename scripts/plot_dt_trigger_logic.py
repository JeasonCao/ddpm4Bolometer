"""
Single fully-annotated diagram explaining the BLIND DT trigger DECISION
RULE itself (not a statistical example) — one clean, promptly-triggering
event, with the rolling baseline window, threshold formula, the n_above
consecutive-sample requirement, and the trigger point all labeled directly
on the waveform. The true onset is shown only as a dotted reference line —
it plays no role in the decision, only in choosing which (of possibly
several) blind triggers to highlight for this illustration.

Usage
-----
python scripts/plot_dt_trigger_logic.py \
    --eval_file /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p2mV/eval_efficiency_10keV.h5 \
    --event_idx 4 \
    --output results/plots/trigger/trigger_logic_diagram.png
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


def run(eval_file, event_idx, output, nsigma, n_above, window_ms, dead_ms,
       match_pre_ms, match_post_ms, kind):
    with h5py.File(eval_file, 'r') as f:
        energy_kev = float(f.attrs['energy_kev'])
        fs         = float(f.attrs['f_sample'])
        onset_s    = float(f.attrs['onset_s'])
        x = f[f'waveforms_{kind}'][event_idx]

    onset_idx = int(round(onset_s * fs))
    window_n  = int(round(window_ms * 1e-3 * fs))
    dead_n    = int(round(dead_ms * 1e-3 * fs))
    match_pre_n  = int(round(match_pre_ms * 1e-3 * fs))
    match_post_n = int(round(match_post_ms * 1e-3 * fs))

    trigger_idxs, mu, rms, threshold = dt_trigger_blind(
        x, fs, window_n, nsigma=nsigma, n_above=n_above, dead_samples=dead_n)

    in_match = (trigger_idxs >= onset_idx - match_pre_n) & (trigger_idxs <= onset_idx + match_post_n)
    assert in_match.any(), (
        'Chosen event has no trigger matched to the true onset — pick another --event_idx. '
        f'(blind triggers found at: {(trigger_idxs / fs * 1e3).tolist()} ms)')
    trigger_idx = int(trigger_idxs[in_match][0])

    dv = np.diff(x)
    t  = np.arange(len(x)) / fs

    # The n_above qualifying samples, in ORIGINAL sample-index terms, are
    # [trigger_idx, trigger_idx + n_above - 1]; dv value "at" original index
    # j is dv[j-1] (dv[k] = x[k+1]-x[k]).
    run_orig_idx = np.arange(trigger_idx, trigger_idx + n_above)
    run_dv       = dv[run_orig_idx - 1]
    # threshold[k] applies to dv[window_n+k] <-> original index window_n+k+1
    thr_at_trigger = float(threshold[trigger_idx - window_n - 1])
    mu_at_trigger  = float(mu[trigger_idx - window_n - 1])
    rms_at_trigger = float(rms[trigger_idx - window_n - 1])
    assert np.all(run_dv > thr_at_trigger), 'Sanity check failed: run not all above threshold'

    # ── zoom windows ──────────────────────────────────────────────────────
    zoom_lo_ms, zoom_hi_ms = -50.0, 120.0
    zoom_lo = max(0, onset_idx + int(zoom_lo_ms * 1e-3 * fs))
    zoom_hi = min(len(x), onset_idx + int(zoom_hi_ms * 1e-3 * fs))

    inset_pad = 4
    inset_lo  = max(0, trigger_idx - inset_pad)
    inset_hi  = min(len(dv), trigger_idx + n_above + inset_pad)

    color = 'tomato' if kind == 'denoised' else 'steelblue'

    fig = plt.figure(figsize=(11, 10.5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.3, 1.3, 1.1], hspace=0.5)
    ax_wave = fig.add_subplot(gs[0])
    ax_dv   = fig.add_subplot(gs[1], sharex=ax_wave)
    ax_zoom = fig.add_subplot(gs[2])

    onset_ms = onset_s * 1e3
    trig_ms  = trigger_idx / fs * 1e3 - onset_ms
    rel_ms   = t * 1e3 - onset_ms

    # ── Panel A: waveform ─────────────────────────────────────────────────
    ax_wave.plot(rel_ms[zoom_lo:zoom_hi], x[zoom_lo:zoom_hi] * 1e3, color=color, lw=1.3)
    ax_wave.axvspan(-match_pre_ms, match_post_ms, color='limegreen', alpha=0.10,
                    label='match window (scoring only)')
    ax_wave.axvline(0, color='gray', ls=':', lw=1.2)
    ax_wave.axvline(trig_ms, color='limegreen', lw=1.8)
    ax_wave.annotate('true onset\n(reference only —\nnot used by the trigger)',
                     xy=(0, 0), xycoords=('data', 'axes fraction'),
                     xytext=(-6, 5), textcoords='offset points', fontsize=8,
                     color='gray', va='bottom', ha='right')
    ax_wave.annotate(f'trigger fires\nt = onset {"+" if trig_ms>=0 else ""}{trig_ms:.0f} ms',
                     xy=(trig_ms, 1), xycoords=('data', 'axes fraction'),
                     xytext=(5, -3), textcoords='offset points', fontsize=8.5,
                     color='darkgreen', va='top', fontweight='bold')
    ax_wave.set_xlim(zoom_lo_ms, zoom_hi_ms)
    ax_wave.set_ylabel('Amplitude [mV]', fontsize=9.5)
    ax_wave.set_title(f'Blind DT Trigger Decision Rule — {energy_kev:.0f} keV, {kind} waveform (event #{event_idx})',
                      fontsize=12)
    ax_wave.tick_params(labelsize=8)
    ax_wave.legend(fontsize=7.5, loc='upper left')
    ax_wave.grid(alpha=0.3)

    # ── Panel B: derivative + rolling threshold ───────────────────────────
    dv_rel_ms = rel_ms[1:]
    ax_dv.plot(dv_rel_ms[zoom_lo:zoom_hi - 1], dv[zoom_lo:zoom_hi - 1] * 1e3,
              color=color, lw=1.3)
    # rolling threshold curve over the zoom range
    thr_lo = max(zoom_lo - window_n - 1, 0)
    thr_hi = max(zoom_hi - 1 - window_n, 0)
    thr_orig_idx = np.arange(thr_lo, thr_hi) + window_n + 1
    thr_rel_ms = thr_orig_idx / fs * 1e3 - onset_ms
    ax_dv.plot(thr_rel_ms, threshold[thr_lo:thr_hi] * 1e3, color='black', ls='--', lw=1.1,
              label='rolling threshold')
    ax_dv.axvline(0, color='gray', ls=':', lw=1.2)
    ax_dv.axvline(trig_ms, color='limegreen', lw=1.8)
    ax_dv.annotate(
        r'threshold(t) $= \mu_{roll}(t) + n_\sigma \cdot \sigma_{roll}(t)$' + '\n'
        rf'at trigger: $= {mu_at_trigger*1e3:.4f} + {nsigma:.0f}\times{rms_at_trigger*1e3:.4f}$ mV/sample'
        '\n(recomputed from the preceding '
        f'{window_ms:.0f} ms at every point — not a fixed line)',
        xy=(zoom_hi_ms * 0.5, thr_at_trigger * 1e3), fontsize=8, color='black',
        xytext=(zoom_hi_ms * 0.22, thr_at_trigger * 1e3 * 1.6 + 0.02), va='bottom')
    ax_dv.set_xlim(zoom_lo_ms, zoom_hi_ms)
    ax_dv.set_ylabel('dV [mV/sample]', fontsize=9.5)
    ax_dv.set_xlabel('Time relative to true onset [ms]', fontsize=9.5)
    ax_dv.tick_params(labelsize=8)
    ax_dv.legend(fontsize=7.5, loc='upper left')
    ax_dv.grid(alpha=0.3)

    # Bracket over the qualifying run on panel B
    run_t = run_orig_idx / fs * 1e3 - onset_ms
    y_bracket = float(np.max(run_dv) * 1e3) * 1.3 + 0.01
    ax_dv.annotate('', xy=(run_t[0], y_bracket), xytext=(run_t[-1], y_bracket),
                  arrowprops=dict(arrowstyle='|-|', color='darkorange', lw=1.6))
    ax_dv.text(run_t.mean(), y_bracket, f'\nn_above = {n_above} consecutive\nsamples > threshold',
              ha='center', va='bottom', fontsize=8.5, color='darkorange', fontweight='bold')
    ax_dv.set_ylim(top=y_bracket * 1.8)

    # ── Panel C: tight zoom on the qualifying run ────────────────────────
    zi = np.arange(inset_lo, inset_hi)
    zi_t = (zi + 1) / fs * 1e3 - onset_ms  # dv[i] <-> original index i+1
    in_run = (zi >= trigger_idx - 1) & (zi < trigger_idx - 1 + n_above)
    zi_thr = threshold[zi - window_n] * 1e3   # threshold at each zoomed dv point

    ax_zoom.plot(zi_t, dv[zi] * 1e3, color='lightgray', lw=1.2, zorder=1)
    ax_zoom.plot(zi_t, zi_thr, color='black', ls='--', lw=1.0, zorder=2, label='rolling threshold')
    ax_zoom.scatter(zi_t[~in_run], dv[zi][~in_run] * 1e3, color='gray', s=40, zorder=3, label='below threshold')
    ax_zoom.scatter(zi_t[in_run], dv[zi][in_run] * 1e3, color='darkorange', s=55, zorder=4,
                   edgecolors='black', linewidth=0.6, label='above threshold (counted)')
    ax_zoom.axvline(trig_ms, color='limegreen', lw=1.6)

    y0, y1 = ax_zoom.get_ylim()
    bracket_y = y1 - 0.05 * (y1 - y0)
    ax_zoom.annotate('', xy=(zi_t[in_run][0], bracket_y), xytext=(zi_t[in_run][-1], bracket_y),
                    arrowprops=dict(arrowstyle='|-|', color='darkorange', lw=1.6))
    ax_zoom.text(zi_t[in_run].mean(), bracket_y, f'\n{n_above} consecutive samples',
                ha='center', va='bottom', fontsize=8.5, color='darkorange', fontweight='bold')

    ax_zoom.set_title('Zoom: the exact n_above-sample run that fires the trigger', fontsize=10)
    ax_zoom.set_xlabel('Time relative to true onset [ms]', fontsize=9.5)
    ax_zoom.set_ylabel('dV [mV/sample]', fontsize=9.5)
    ax_zoom.tick_params(labelsize=8)
    ax_zoom.legend(fontsize=7.5, loc='lower right')
    ax_zoom.grid(alpha=0.3)

    output = timestamped_path(output)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {output}')
    print(f'  all blind triggers found (ms after onset): {(trigger_idxs/fs*1e3 - onset_ms).tolist()}')
    print(f'  shown trigger_idx={trigger_idx}  t={trig_ms:.1f} ms after onset')
    print(f'  threshold@trigger={thr_at_trigger:.6f} V  mu={mu_at_trigger:.6f} V  rms={rms_at_trigger:.6f} V')


def main():
    parser = argparse.ArgumentParser(description='Single annotated blind DT trigger logic diagram')
    parser.add_argument('--eval_file', required=True)
    parser.add_argument('--event_idx', type=int, default=4)
    parser.add_argument('--kind', choices=['noisy', 'denoised'], default='denoised')
    parser.add_argument('--nsigma',  type=float, default=4.0)
    parser.add_argument('--n_above', type=int,   default=5)
    parser.add_argument('--window_ms', type=float, default=ROLLING_WINDOW_MS)
    parser.add_argument('--dead_ms',   type=float, default=DEAD_TIME_MS)
    parser.add_argument('--match_pre_ms',  type=float, default=MATCH_PRE_MS)
    parser.add_argument('--match_post_ms', type=float, default=MATCH_POST_MS)
    parser.add_argument('--output', default=os.path.join(_RESULTS_DIR, 'plots', 'trigger', 'trigger_logic_diagram.png'))
    args = parser.parse_args()

    run(args.eval_file, args.event_idx, args.output, args.nsigma, args.n_above,
       args.window_ms, args.dead_ms, args.match_pre_ms, args.match_post_ms, args.kind)


if __name__ == '__main__':
    main()
