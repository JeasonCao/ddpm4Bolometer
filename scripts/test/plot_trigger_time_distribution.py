"""
Diagnostic: distribution of blind DT trigger times (relative to true onset),
aggregated across many events/energies.

Two figures:
  1. ALL trigger times found (matched + extra) — shows how dominant the
     real-pulse peak is relative to background.
  2. Only the "extra" (unmatched) trigger times, i.e. with the correctly-
     matched ones cut out — isolates the background/dark population and
     the near-onset "early leak" effect found earlier in the denoised data.

Also reports how often MULTIPLE blind triggers fire within a single event's
10s trace (matched + extra combined).

This is an exploratory/one-off script (recomputes the blind trigger directly
from eval HDF5 files rather than depending on run_dt_trigger.py's saved
outputs) — kept in scripts/test/ rather than the main scripts/ directory.

Usage
-----
python scripts/test/plot_trigger_time_distribution.py \
    --eval_dir /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_7p5mV \
    --kind denoised \
    --output_dir results/plots/trigger/diagnostics
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.basics.trigger import dt_trigger_blind
from src.basics.plot_utils import timestamped_path

ROLLING_WINDOW_MS = 300.0
DEAD_TIME_MS      = 300.0
MATCH_PRE_MS  = 100.0
MATCH_POST_MS = 200.0
BIN_WIDTH_MS  = 10.0


def collect_times(eval_dir, kind, nsigma, n_above, window_ms, dead_ms,
                  match_pre_ms, match_post_ms):
    files = sorted(glob.glob(os.path.join(eval_dir, 'eval_efficiency_*.h5')))
    if not files:
        raise FileNotFoundError(f'No eval_efficiency_*.h5 files in {eval_dir}')

    all_times_ms   = []   # every blind trigger, relative to its own event's onset
    extra_times_ms = []   # subset not matched to the true onset
    n_triggers_all = []   # per-event trigger COUNT (matched + extra)

    for fp in files:
        with h5py.File(fp, 'r') as f:
            fs      = float(f.attrs['f_sample'])
            onset_s = float(f.attrs['onset_s'])
            waveforms = f[f'waveforms_{kind}'][:]

        onset_idx = int(round(onset_s * fs))
        window_n  = int(round(window_ms * 1e-3 * fs))
        dead_n    = int(round(dead_ms * 1e-3 * fs))
        match_pre_n  = int(round(match_pre_ms * 1e-3 * fs))
        match_post_n = int(round(match_post_ms * 1e-3 * fs))
        lo = onset_idx - match_pre_n
        hi = onset_idx + match_post_n

        for i in range(waveforms.shape[0]):
            idxs, _, _, _ = dt_trigger_blind(
                waveforms[i], fs, window_n, nsigma=nsigma, n_above=n_above,
                dead_samples=dead_n)
            n_triggers_all.append(len(idxs))
            if len(idxs) == 0:
                continue
            t_ms = idxs / fs * 1e3 - onset_s * 1e3
            all_times_ms.extend(t_ms.tolist())
            in_match = (idxs >= lo) & (idxs <= hi)
            extra_times_ms.extend(t_ms[~in_match].tolist())

        print(f'  {os.path.basename(fp)}: {waveforms.shape[0]} events processed')

    return np.array(all_times_ms), np.array(extra_times_ms), np.array(n_triggers_all)


def plot_all(times_ms, match_pre_ms, match_post_ms, title, output):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lo, hi = times_ms.min(), times_ms.max()
    bins = np.arange(np.floor(lo / BIN_WIDTH_MS) * BIN_WIDTH_MS,
                     np.ceil(hi / BIN_WIDTH_MS) * BIN_WIDTH_MS + BIN_WIDTH_MS, BIN_WIDTH_MS)
    ax.hist(times_ms, bins=bins, color='steelblue', alpha=0.85)
    ax.axvline(0, color='gray', ls=':', lw=1.2, label='true onset')
    ax.axvspan(-match_pre_ms, match_post_ms, color='limegreen', alpha=0.12, label='match window')
    ax.set_yscale('log')
    ax.set_xlabel('Trigger time relative to true onset [ms]', fontsize=10)
    ax.set_ylabel(f'Count per {BIN_WIDTH_MS:.0f} ms bin (log scale)', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    output = timestamped_path(output)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {output}')


def plot_extra(times_ms, match_pre_ms, match_post_ms, title, output):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if len(times_ms) == 0:
        ax.text(0.5, 0.5, 'No unmatched triggers found', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, color='gray')
    else:
        lo, hi = times_ms.min(), times_ms.max()
        bins = np.arange(np.floor(lo / BIN_WIDTH_MS) * BIN_WIDTH_MS,
                         np.ceil(hi / BIN_WIDTH_MS) * BIN_WIDTH_MS + BIN_WIDTH_MS, BIN_WIDTH_MS)
        ax.hist(times_ms, bins=bins, color='darkorange', alpha=0.85)
    ax.axvline(0, color='gray', ls=':', lw=1.2, label='true onset')
    ax.axvspan(-match_pre_ms, match_post_ms, color='limegreen', alpha=0.12, label='match window (excluded)')
    ax.set_yscale('log')
    ax.set_xlabel('Trigger time relative to true onset [ms]', fontsize=10)
    ax.set_ylabel(f'Count per {BIN_WIDTH_MS:.0f} ms bin (log scale)', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    output = timestamped_path(output)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {output}')


def main():
    parser = argparse.ArgumentParser(description='Blind DT trigger time distribution diagnostics')
    parser.add_argument('--eval_dir', required=True,
                        help='Directory of eval_efficiency_*.h5 files (one noise level)')
    parser.add_argument('--kind', choices=['noisy', 'denoised'], default='denoised')
    parser.add_argument('--nsigma',  type=float, default=4.0)
    parser.add_argument('--n_above', type=int,   default=5)
    parser.add_argument('--window_ms', type=float, default=ROLLING_WINDOW_MS)
    parser.add_argument('--dead_ms',   type=float, default=DEAD_TIME_MS)
    parser.add_argument('--match_pre_ms',  type=float, default=MATCH_PRE_MS)
    parser.add_argument('--match_post_ms', type=float, default=MATCH_POST_MS)
    parser.add_argument('--output_dir', default=os.path.join(_RESULTS_DIR, 'plots', 'trigger', 'diagnostics'))
    args = parser.parse_args()

    ds_tag = os.path.basename(args.eval_dir.rstrip('/'))
    print(f'Scanning {ds_tag} ({args.kind}) ...')
    all_times, extra_times, n_triggers = collect_times(
        args.eval_dir, args.kind, args.nsigma, args.n_above,
        args.window_ms, args.dead_ms, args.match_pre_ms, args.match_post_ms)

    print(f'Total blind triggers: {len(all_times)}  |  unmatched ("extra"): {len(extra_times)}')

    vals, counts = np.unique(n_triggers, return_counts=True)
    n_events = len(n_triggers)
    print(f'Per-event trigger count distribution (n={n_events} events):')
    for v, c in zip(vals, counts):
        print(f'    {v} trigger(s): {c} events ({c/n_events*100:.2f}%)')
    print(f'  -> events with >=2 triggers: {(n_triggers>=2).sum()} ({(n_triggers>=2).mean()*100:.2f}%)')

    plot_all(all_times, args.match_pre_ms, args.match_post_ms,
            f'All blind trigger times — {ds_tag}, {args.kind}\n'
            f'(nsigma={args.nsigma}, n_above={args.n_above}, rolling window={args.window_ms:.0f} ms)',
            os.path.join(args.output_dir, f'trigger_times_all_{ds_tag}_{args.kind}.png'))

    plot_extra(extra_times, args.match_pre_ms, args.match_post_ms,
              f'Unmatched ("extra") trigger times only — {ds_tag}, {args.kind}\n'
              f'(correctly-matched triggers within [-{args.match_pre_ms:.0f}, +{args.match_post_ms:.0f}] ms cut out)',
              os.path.join(args.output_dir, f'trigger_times_extra_{ds_tag}_{args.kind}.png'))


if __name__ == '__main__':
    main()
