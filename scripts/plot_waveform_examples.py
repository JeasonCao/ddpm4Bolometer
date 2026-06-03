"""
Plot time-domain waveform examples: noisy vs denoised, with fit overlay.

Picks events at low / mid / high SNR from one energy point and shows
  - raw noisy waveform + tri-exp fit
  - denoised waveform + tri-exp fit
  - clean template (truth)

Usage
-----
python scripts/plot_waveform_examples.py \
    --eval_file /home/wsl_0vbb/DDPM4bolometer/eval_data/resolution/eval_1461keV.h5 \
    --output    /home/wsl_0vbb/DDPM4bolometer/plots/waveform_examples_1461keV.png
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

from src.basics.fit import fit_pulse

F_SAMPLE = 1000.0


def pick_events(snr_db, n=3):
    """Pick indices near 25th, 50th, 75th percentile of SNR distribution."""
    pcts = [20, 50, 80]
    thresholds = np.percentile(snr_db, pcts)
    indices = []
    for th in thresholds:
        idx = int(np.argmin(np.abs(snr_db - th)))
        indices.append(idx)
    return indices, thresholds


def reconstruct_fit(waveform, fs=F_SAMPLE):
    """Return (t, model_curve, fr) or None on failure."""
    fr = fit_pulse(waveform, n_pulses=1, fs=fs)
    if not fr.success or fr.model_fn is None:
        return None
    t = np.arange(len(waveform)) / fs
    model = fr.model_fn(t, *fr.params)
    return t, model, fr


def plot_waveform_examples(eval_file: str, output: str):
    with h5py.File(eval_file, 'r') as f:
        energy_kev   = float(f.attrs['energy_kev'])
        snr_db       = f['snr_db'][:]
        w_noisy      = f['waveforms_noisy'][:]
        w_denoised   = f['waveforms_denoised'][:]
        clean_tmpl   = f['clean_template'][:]

    indices, snr_vals = pick_events(snr_db)
    t_tmpl = np.arange(len(clean_tmpl)) / F_SAMPLE

    n_events = len(indices)
    fig, axes = plt.subplots(n_events, 2, figsize=(14, 4 * n_events), sharex=True)
    fig.suptitle(f'Waveform Examples — {energy_kev:.0f} keV', fontsize=13)

    col_labels = ['Before denoising', 'After denoising']
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=11)

    for row, (idx, snr) in enumerate(zip(indices, snr_vals)):
        wn = w_noisy[idx]
        wd = w_denoised[idx]
        t  = np.arange(len(wn)) / F_SAMPLE

        for col, (wf, color) in enumerate([(wn, 'steelblue'), (wd, 'tomato')]):
            ax = axes[row, col]
            ax.plot(t, wf, color=color, alpha=0.6, lw=0.8, label='Waveform')
            ax.plot(t_tmpl, clean_tmpl, 'k--', lw=1.0, alpha=0.5, label='Clean template')

            res = reconstruct_fit(wf)
            if res is not None:
                t_fit, model, fr = res
                amp = fr.peak_amps[0] if fr.peak_amps else float('nan')
                chi2 = fr.chi2_per_ndf
                ax.plot(t_fit, model, 'm-', lw=1.5,
                        label=f'Fit  A={amp:.3f} V  χ²/ndf={chi2:.2f}')

            ax.set_ylabel('Amplitude (V)', fontsize=9)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f'{col_labels[col]}', fontsize=11)
            ax.text(0.02, 0.95, f'SNR={snr:.1f} dB',
                    transform=ax.transAxes, fontsize=9, va='top')

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
    parser.add_argument('--output',    default=None,
                        help='Output path; defaults to results/plots/waveform_examples_<energy>.png')
    args = parser.parse_args()
    output = args.output
    if output is None:
        stem = os.path.splitext(os.path.basename(args.eval_file))[0]
        output = os.path.join(_RESULTS_DIR, 'plots', f'waveform_examples_{stem[5:]}.png')
    plot_waveform_examples(args.eval_file, output)


if __name__ == '__main__':
    main()
