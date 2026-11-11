"""
Plot time-domain waveform examples: noisy vs denoised, with fit overlay.

Picks events near the 20th / 50th / 80th SNR percentile and shows
  - raw waveform + tri-exp fit (A, chi²/ndf) for each side
  - true noiseless pulse overlaid for reference

Usage
-----
python scripts/plot_waveform_examples.py \
    --eval_file /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/resolution/eval_1461keV.h5 \
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
from src.basics.waveform_utils import plot_waveform_pair, F_SAMPLE


def pick_events(snr_db):
    """Return indices near the 20th / 50th / 80th SNR percentiles."""
    thresholds = np.percentile(snr_db, [20, 50, 80])
    return [int(np.argmin(np.abs(snr_db - th))) for th in thresholds], thresholds


def plot_waveform_examples(eval_file: str, output: str):
    with h5py.File(eval_file, 'r') as f:
        energy_kev   = float(f.attrs['energy_kev'])
        snr_db       = f['snr_db'][:]
        w_noisy      = f['waveforms_noisy'][:]
        w_denoised   = f['waveforms_denoised'][:]
        clean_tmpl   = f['clean_template'][:]

    indices, snr_vals = pick_events(snr_db)
    n = len(indices)

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), sharex=True, squeeze=False)
    fig.suptitle(f'Waveform Examples — {energy_kev:.0f} keV', fontsize=13)
    axes[0, 0].set_title('Before denoising', fontsize=11)
    axes[0, 1].set_title('After denoising',  fontsize=11)

    for row, (idx, snr) in enumerate(zip(indices, snr_vals)):
        plot_waveform_pair(
            axes[row, 0], axes[row, 1],
            w_noisy[idx], w_denoised[idx],
            snr_db=snr,
            event_label=f'event #{idx}',
            clean_template=clean_tmpl,
        )

    for col in range(2):
        axes[-1, col].set_xlabel('Time (s)', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', required=True)
    parser.add_argument('--output',    default=None)
    parser.add_argument('--suffix',    default='')
    parser.add_argument('--model_cfg', default=None, metavar='MODEL_DIR')
    args = parser.parse_args()
    suffix = suffix_from_model_dir(args.model_cfg) if args.model_cfg else args.suffix
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.eval_file))[0]
        args.output = os.path.join(
            _RESULTS_DIR, 'plots', 'resolution',
            f'waveform_examples_{stem[5:]}{suffix}.png')
    plot_waveform_examples(args.eval_file, args.output)


if __name__ == '__main__':
    main()
