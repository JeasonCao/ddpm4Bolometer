"""
Plot waveform comparisons (before/after denoising) for efficiency datasets.

Two modes per energy point:
  --mode failed  : events where fit converged before denoising but NOT after (up to 10)
  --mode random  : 10 randomly chosen events

Each figure: one row per event, 2 columns (before | after).
Each panel shows: raw waveform + tri-exp fit (A, chi²/ndf) + true noiseless pulse.

Usage
-----
python scripts/plot_efficiency_waveforms.py \
    --eval_dir  /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p2mV \
    --fit_dir   results/fit_results/efficiency_1p2mV \
    --mode      failed \
    --model_cfg /mnt/d/storage/research/DDPM4bolometer/model_output_cloud
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

from src.basics.plot_utils import suffix_from_model_dir
from src.basics.waveform_utils import plot_waveform_pair

MAX_EVENTS = 10
RNG_SEED   = 0


def plot_events(wf_noisy, wf_denoised, clean_template, snr_db,
                indices, energy_kev, title, output):
    n = len(indices)
    if n == 0:
        print(f"  No events for: {title}")
        return

    fig, axes = plt.subplots(n, 2, figsize=(14, 3.8 * n), squeeze=False)
    fig.suptitle(f'{title}  —  {energy_kev:.0f} keV', fontsize=13)
    axes[0, 0].set_title('Before denoising', fontsize=11)
    axes[0, 1].set_title('After denoising',  fontsize=11)

    for row, idx in enumerate(indices):
        plot_waveform_pair(
            axes[row, 0], axes[row, 1],
            wf_noisy[idx], wf_denoised[idx],
            snr_db=float(snr_db[idx]),
            event_label=f'event #{idx}',
            clean_template=clean_template,
        )

    for col in range(2):
        axes[-1, col].set_xlabel('Time (s)', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {output}")


def process_one_file(eval_path, fit_path, mode, out_dir, suffix, dataset_tag):
    with h5py.File(eval_path, 'r') as ef:
        energy_kev     = float(ef.attrs['energy_kev'])
        wf_noisy       = ef['waveforms_noisy'][:]
        wf_denoised    = ef['waveforms_denoised'][:]
        clean_template = ef['clean_template'][:]
        snr_db         = ef['snr_db'][:]

    with h5py.File(fit_path, 'r') as ff:
        conv_noisy    = ff['converged_noisy'][:].astype(bool)
        conv_denoised = ff['converged_denoised'][:].astype(bool)

    energy_tag = f'{energy_kev:.0f}keV'
    rng = np.random.default_rng(RNG_SEED)

    if mode == 'failed':
        candidates = np.where(conv_noisy & ~conv_denoised)[0]
        indices = np.sort(
            candidates if len(candidates) <= MAX_EVENTS
            else rng.choice(candidates, MAX_EVENTS, replace=False)
        )
        title = f'Failed after denoising (converged before, not after)  [{dataset_tag}]'
        fname = f'failed_{energy_tag}_{dataset_tag}{suffix}.png'
    else:
        indices = np.sort(rng.choice(len(snr_db), min(MAX_EVENTS, len(snr_db)), replace=False))
        title = f'Random events  [{dataset_tag}]'
        fname = f'random_{energy_tag}_{dataset_tag}{suffix}.png'

    print(f"  {energy_tag}: {len(indices)} events ({mode})")
    plot_events(wf_noisy, wf_denoised, clean_template, snr_db,
                indices, energy_kev, title, os.path.join(out_dir, fname))


def run(eval_dir, fit_dir, mode, out_dir, suffix):
    dataset_tag = os.path.basename(eval_dir)
    for fname in sorted(f for f in os.listdir(eval_dir) if f.endswith('.h5')):
        eval_path = os.path.join(eval_dir, fname)
        fit_path  = os.path.join(fit_dir, fname.replace('eval_', 'fit_'))
        if not os.path.isfile(fit_path):
            print(f"  WARNING: no fit file for {fname}, skipping")
            continue
        process_one_file(eval_path, fit_path, mode, out_dir, suffix, dataset_tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir',   required=True)
    parser.add_argument('--fit_dir',    required=True)
    parser.add_argument('--mode',       choices=['failed', 'random'], default='failed')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--suffix',     default='')
    parser.add_argument('--model_cfg',  default=None, metavar='MODEL_DIR')
    args = parser.parse_args()

    suffix  = suffix_from_model_dir(args.model_cfg) if args.model_cfg else args.suffix
    out_dir = args.output_dir or os.path.join(_RESULTS_DIR, 'plots', 'efficiency_waveforms')

    print(f"Mode: {args.mode} | dataset: {os.path.basename(args.eval_dir)}")
    run(args.eval_dir, args.fit_dir, args.mode, out_dir, suffix)


if __name__ == '__main__':
    main()
